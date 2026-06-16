"""Grounding relevance: reject weak matches; return only the top doc's chunks.

Fixtures mirror the real AI Core document-grounding response shape:
  body.results[0].results[0].dataRepository.documents[].chunks[]
    .content
    .searchScores.aggregatedScore.value   (0..1 cosine similarity)
  document.metadata[] = [{key, value:[...]}]   (title, filePath, webUrl)

The real bug: query "partner function SP missing" returned AIF_003 vendor_not_found
with match:true because ANY returned doc was accepted. Its best chunk scored only
0.3908 — below a sensible relevance floor.
"""
from app.agent import _select_grounding


def _chunk(content, score):
    return {"content": content, "searchScores": {"aggregatedScore": {"value": score}}}


def _doc(title, chunks):
    return {
        "metadata": [
            {"key": "title", "value": [title]},
            {"key": "filePath", "value": [f"jouledemo/{title}"]},
            {"key": "webUrl", "value": [f"https://x/{title}"]},
        ],
        "chunks": chunks,
    }


def _body(docs):
    return {"results": [{"results": [{"dataRepository": {"documents": docs}}]}]}


# Reproduces the reported false positive: best chunk 0.3908, all weakly related.
_BOGUS = _body([
    _doc("AIF_003_vendor_not_found.txt", [_chunk("vendor not found A", 0.3908),
                                          _chunk("vendor not found B", 0.3591)]),
    _doc("sap_error_codes_dataset.txt", [_chunk("F5/567 dup doc", 0.3254)]),
    _doc("AIF_099_generic_check_error.txt", [_chunk("generic check", 0.3140)]),
])

# A genuine hit: the right doc scores well above the floor.
_GOOD = _body([
    _doc("AIF_003_vendor_not_found.txt", [_chunk("root cause", 0.74),
                                          _chunk("resolution steps", 0.61)]),
    _doc("sap_error_codes_dataset.txt", [_chunk("unrelated", 0.30)]),
])

# Real "resolution steps for error KI/260" response (abridged to the scores that
# matter): the catalog doc with KI/260 tops at 0.5204; its real resolution-step
# chunks score 0.42/0.41 (lower than a generic "Escalation" chunk). The two
# unrelated docs peak at 0.44 and 0.40. Best overall = 0.5204 -> accept at 0.45.
_KI260_REAL = _body([
    _doc("sap_error_codes_dataset.txt", [
        _chunk("## Escalation ... escalate with the following information", 0.5204358920711458),
        _chunk("## Error Index ... KI/260 Cost Center Blocked", 0.41975482920526747),
        _chunk("Step 7 Reprocess ... KI/260 ... Restart", 0.416473796690277),
        _chunk("Step 4 document type mapping", 0.3976725082343407),
    ]),
    _doc("AIF_099_generic_check_error.txt", [_chunk("Resolution by cause", 0.4442964935354917)]),
    _doc("AIF_003_vendor_not_found.txt", [_chunk("Resolution steps vendor", 0.39965478961428136)]),
])


def test_real_ki260_query_accepted_at_default_floor():
    # Validates the 0.45 default against real scores: good=0.52 accepted,
    # and the matched doc is the one actually containing KI/260.
    r = _select_grounding(_KI260_REAL, min_score=0.45)
    assert r["match"] is True
    assert round(r["bestScore"], 4) == 0.5204
    assert r["groundingSource"]["title"] == "sap_error_codes_dataset.txt"
    # the real KI/260 resolution content is present even though it scored below
    # the generic "Escalation" chunk (we include the whole matched doc).
    assert "KI/260" in r["groundingText"]
    assert "Reprocess" in r["groundingText"]


def test_default_floor_separates_real_good_from_bogus():
    from app.agent import AICORE_GROUNDING_MIN_SCORE
    assert _select_grounding(_KI260_REAL, AICORE_GROUNDING_MIN_SCORE)["match"] is True   # 0.52
    assert _select_grounding(_BOGUS, AICORE_GROUNDING_MIN_SCORE)["match"] is False       # 0.39


def test_weak_match_is_rejected():
    r = _select_grounding(_BOGUS, min_score=0.45)
    assert r["match"] is False
    assert round(r["bestScore"], 4) == 0.3908   # surfaced for observability


def test_strong_match_accepted_with_top_doc_only():
    r = _select_grounding(_GOOD, min_score=0.45)
    assert r["match"] is True
    assert round(r["bestScore"], 2) == 0.74
    assert r["groundingSource"]["title"] == "AIF_003_vendor_not_found.txt"
    # groundingText carries the matched doc's chunks (best-first), NOT other docs.
    assert "root cause" in r["groundingText"]
    assert "resolution steps" in r["groundingText"]
    assert "unrelated" not in r["groundingText"]


def test_picks_highest_scoring_document_not_first():
    # Top-scoring doc is the SECOND in the list — must still be chosen.
    body = _body([
        _doc("low.txt", [_chunk("low", 0.50)]),
        _doc("high.txt", [_chunk("high", 0.80)]),
    ])
    r = _select_grounding(body, min_score=0.45)
    assert r["match"] is True
    assert r["groundingSource"]["title"] == "high.txt"


def test_empty_response_is_no_match():
    assert _select_grounding({"results": []}, min_score=0.45)["match"] is False
    assert _select_grounding(_body([]), min_score=0.45)["match"] is False


def test_chunks_without_scores_do_not_crash():
    body = _body([_doc("x.txt", [{"content": "no score here"}])])
    r = _select_grounding(body, min_score=0.45)
    assert r["match"] is False   # unscored chunk treated as 0
