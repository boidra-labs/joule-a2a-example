"""Parse the AI Core Orchestration /completion response into a grounding result.

The orchestration response embeds the grounded answer as a JSON STRING in
final_result.choices[0].message.content, with {id, root_cause, resolution_step[]}
(and optionally resolvable_from_references). intermediate_results.grounding.data
carries the raw retrieved text. _parse_orchestration_grounding turns this into the
same shape the analysis flow expects: {match, rootCause, resolutionSteps[], ...}.
"""
import json

from app.agent import _parse_orchestration_grounding, _split_error_query


def test_split_error_query_id_number_message():
    assert _split_error_query("AIF/099 Processing terminated") == (
        "AIF", "099", "Processing terminated")


def test_split_error_query_no_text():
    assert _split_error_query("FI/042") == ("FI", "042", "")


def test_split_error_query_no_slash():
    # falls back: whole token as id, empty number
    assert _split_error_query("SOMECODE only text") == ("SOMECODE", "", "only text")

_CONTENT = {
    "id": "AIF-099",
    "root_cause": "The combination with 099 (+/-) is not supported per the reference.",
    "resolution_step": [
        "Review the source data to find why a 099 (+/-) combination was generated.",
        "Correct the source transaction to a supported combination.",
        "Reprocess the corrected data through the interface.",
    ],
}


def _resp(content_obj, resolvable=True, grounding_text="ref material here"):
    c = dict(content_obj)
    c["resolvable_from_references"] = resolvable
    return {
        "intermediate_results": {
            "grounding": {"data": {"grounding_query": "AIF/099 ...",
                                   "grounding_result": grounding_text}}
        },
        "final_result": {
            "choices": [{"message": {"role": "assistant",
                                     "content": json.dumps(c)}}]
        },
    }


def test_parses_root_cause_and_steps():
    r = _parse_orchestration_grounding(_resp(_CONTENT))
    assert r["match"] is True
    assert r["rootCause"] == _CONTENT["root_cause"]
    assert r["resolutionSteps"] == _CONTENT["resolution_step"]
    # the retrieved reference text is preserved for transparency
    assert "ref material here" in r["groundingText"]


def test_not_resolvable_is_no_match():
    r = _parse_orchestration_grounding(_resp(_CONTENT, resolvable=False))
    assert r["match"] is False


def test_handles_content_already_dict():
    # some deployments may return content as an object, not a JSON string
    resp = _resp(_CONTENT)
    resp["final_result"]["choices"][0]["message"]["content"] = dict(
        _CONTENT, resolvable_from_references=True
    )
    r = _parse_orchestration_grounding(resp)
    assert r["match"] is True
    assert r["resolutionSteps"][0].startswith("Review the source data")


def test_malformed_response_is_no_match():
    assert _parse_orchestration_grounding({})["match"] is False
    assert _parse_orchestration_grounding(
        {"final_result": {"choices": [{"message": {"content": "not json"}}]}}
    )["match"] is False


def test_missing_root_cause_is_no_match():
    r = _parse_orchestration_grounding(_resp({"id": "X", "resolution_step": []}))
    assert r["match"] is False
