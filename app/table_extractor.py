from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz


@dataclass
class OCRArtifacts:
    raw_text: str
    markdown_table: str
    diagnostics: Dict[str, Any]


class TableImageAnalyzer:
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, th = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def detect_table_cells(self, bin_img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
        inv = 255 - bin_img
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)
        grid = cv2.add(h_lines, v_lines)

        contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 40 and h > 15:
                boxes.append((x, y, w, h))

        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes, {"detected_boxes": len(boxes)}

    def ocr_cells_to_markdown(self, bin_img: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> str:
        rows: Dict[int, List[Tuple[int, str]]] = {}
        for x, y, w, h in boxes:
            pad = 2
            roi = bin_img[max(0, y + pad): y + h - pad, max(0, x + pad): x + w - pad]
            text = pytesseract.image_to_string(roi, config="--psm 6").strip().replace("\n", " ")
            if not text:
                continue

            row_key = (y // 20) * 20
            rows.setdefault(row_key, []).append((x, text))

        ordered_rows = []
        for rk in sorted(rows.keys()):
            cols = [t for _, t in sorted(rows[rk], key=lambda x: x[0])]
            ordered_rows.append("| " + " | ".join(cols) + " |")
        return "\n".join(ordered_rows)

    def extract(self, image_bytes: bytes) -> OCRArtifacts:
        bin_img = self.preprocess(image_bytes)
        boxes, diagnostics = self.detect_table_cells(bin_img)
        markdown_table = self.ocr_cells_to_markdown(bin_img, boxes)
        raw_text = pytesseract.image_to_string(bin_img, config="--psm 6")
        diagnostics["raw_text_chars"] = len(raw_text)
        diagnostics["markdown_lines"] = len(markdown_table.splitlines()) if markdown_table else 0
        return OCRArtifacts(raw_text=raw_text, markdown_table=markdown_table, diagnostics=diagnostics)


def compare_json_results(pred: List[Dict[str, Any]], expected: List[Dict[str, Any]]) -> tuple[float, str]:
    if not expected:
        return 0.0, "No expected JSON supplied"

    pred_blob = json.dumps(pred, sort_keys=True, ensure_ascii=False)
    exp_blob = json.dumps(expected, sort_keys=True, ensure_ascii=False)
    score = fuzz.token_sort_ratio(pred_blob, exp_blob)

    summary = "match>=90" if score >= 90 else "match<90"
    return float(score), summary
