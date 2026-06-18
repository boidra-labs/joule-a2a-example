"""AIF help grounding: parse a help.sap.com retrieval response into answer text.

get_aif_help_tool grounds general AIF concept/explanation questions (e.g. "what is
an AIF namespace", "explain status 'in process'") against SAP Help, returning
chunk text + the source link for the agent to answer from. The parser reuses the
retrieval response shape; this locks the extraction and the cookbook fallback link.
"""
from app.agent import _select_help_grounding, AIF_COOKBOOK_URL


def _chunk(content, score):
    return {"content": content, "searchScores": {"aggregatedScore": {"value": score}}}


def _doc(title, weburl, chunks):
    return {
        "metadata": [
            {"key": "title", "value": [title]},
            {"key": "webUrl", "value": [weburl]},
        ],
        "chunks": chunks,
    }


def _body(docs):
    return {"results": [{"results": [{"dataRepository": {"documents": docs}}]}]}


def test_extracts_help_text_and_source():
    body = _body([_doc(
        "AIF Interface Monitoring",
        "https://help.sap.com/docs/abapconn/aif/monitoring",
        [_chunk("A namespace groups related AIF interfaces.", 0.71),
         _chunk("The interface monitor shows message status.", 0.66)],
    )])
    r = _select_help_grounding(body, min_score=0.45)
    assert r["match"] is True
    assert "namespace groups related AIF interfaces" in r["helpText"]
    assert r["source"] == "https://help.sap.com/docs/abapconn/aif/monitoring"


def test_weak_match_falls_back_to_cookbook():
    body = _body([_doc("Unrelated", "https://x/y", [_chunk("noise", 0.20)])])
    r = _select_help_grounding(body, min_score=0.45)
    assert r["match"] is False
    # the canonical AIF cookbook link is always available as a fallback source
    assert r["source"] == AIF_COOKBOOK_URL


def test_empty_response_no_match_with_cookbook_source():
    r = _select_help_grounding({"results": []}, min_score=0.45)
    assert r["match"] is False
    assert r["source"] == AIF_COOKBOOK_URL


def test_cookbook_url_is_the_support_content_doc():
    assert AIF_COOKBOOK_URL.startswith("https://help.sap.com/")
    assert "3354079452" in AIF_COOKBOOK_URL
