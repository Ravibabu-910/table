from app.table_extractor import compare_json_results


def test_compare_score_high_for_same_json():
    sample = [{"assessment": "A", "PBO_n10": "1 (10.0)"}]
    score, summary = compare_json_results(sample, sample)
    assert score == 100.0
    assert summary == "match>=90"
