"""The experimental Detect-shaped adapter (kev.detect) against a stub predictor: the mapping, the flag rule, the model string, the limits."""
import re

import pytest

from kev.api import SystemOneRequest
from kev.detect import MAX_DEFINITION_CHARS, MAX_DEFINITIONS, MAX_TEXT_CHARS, QUESTION, detect, flag, floor6, model_string
from kev.errors import KevError
from kev.runtime import Result


class Stub:
    """Answers each Noul question with a fixed probability of yes, looked up by a keyword of its instructions."""
    def __init__(self, yes, identity="a" * 64): self.yes, self._id, self.requests = yes, identity, []

    def identity(self): return self._id

    def answer(self, req: SystemOneRequest):
        self.requests.append(req)
        ps = [[1 - self.yes[q.instructions[len(QUESTION):]], self.yes[q.instructions[len(QUESTION):]]] for q in req.questions.values()]
        return Result({}, ps, 10, 1, 5, 1.0, False)


def body(defs, threshold=None, text="some text"):
    b = {"content": [{"type": "text", "data": text}], "definitions": defs}
    if threshold is not None: b["threshold"] = threshold
    return b


def test_each_definition_is_one_noul_question_over_the_text():
    stub = Stub({"def a": 0.9, "def b": 0.1})
    detect(stub, body({"b": "def b", "a": "def a"}, text="the content"))
    req = stub.requests[0]
    assert req.state == "the content" and list(req.questions) == ["a", "b"]
    assert [(q.type, q.instructions) for q in req.questions.values()] == [("noul", QUESTION + "def a"), ("noul", QUESTION + "def b")]


def test_scores_are_the_probability_of_yes_floored_to_6_decimals_and_flagged_by_the_rule():
    stub = Stub({"d1": 0.9999999, "d2": 0.5, "d3": 0.4999999, "d4": 0.5, "d5": 0.0000004})
    r = detect(stub, body({"c": "d1", "b": "d2", "d": "d3", "a": "d4", "e": "d5"}, 0.5))
    assert r["scores"] == {"a": 0.5, "b": 0.5, "c": 0.999999, "d": 0.499999, "e": 0.0}
    assert r["flagged"] == ["c", "a", "b"]                         # score descending, ties by id; 0.499999 is below the line
    assert set(r) == {"model", "scores", "flagged"} and all(re.fullmatch(r"[a-z][a-z0-9_]*", k) for k in r["scores"])


def test_default_threshold_is_half_and_it_is_not_echoed():
    r = detect(Stub({"x": 0.5, "y": 0.49}), body({"a": "x", "b": "y"}))
    assert r["flagged"] == ["a"] and "threshold" not in r


def test_floor_and_flag_helpers():
    assert floor6(0.29) == 0.29 and floor6(0.1 + 0.2) == 0.3 and floor6(0.9999999) == 0.999999 and floor6(1.0) == 1.0 and floor6(-1e-9) == 0.0
    assert flag({"a": 0.5, "b": 0.6, "c": 0.5}, 0.5) == ["b", "a", "c"]
    assert flag({"a": 0.424743, "b": 0.424742}, 0.424743) == ["a"]
    assert flag({"a": 0.500001}, 0.50000049) == ["a"] and flag({"a": 0.5}, 0.50000049) == []     # a finer threshold counts as the next 6-decimal number up


def test_model_string_is_stable_and_depends_on_checkpoint_and_definitions():
    d = {"a": "first", "b": "second"}
    s = model_string("c" * 64, d)
    assert re.fullmatch(r"kev/v\d+\.\d+(\.\d+)?/[0-9a-f]{16}", s)
    assert s == model_string("c" * 64, {"b": "second", "a": "first"})              # order of definitions does not matter
    assert s != model_string("d" * 64, d) and s != model_string("c" * 64, {"a": "first", "b": "other"}) and s != model_string("c" * 64, {"a": "first"})
    assert detect(Stub({"first": 0.2, "second": 0.2}, "c" * 64), body(d))["model"] == s


@pytest.mark.parametrize("b,code,path", [
    ([], "bad_request", None),
    ({"content": [{"type": "text", "data": "x"}], "definitions": {"a": "b"}, "sensitivity": 1}, "bad_request", "sensitivity"),
    ({"definitions": {"a": "b"}}, "bad_request", "content"),
    ({"content": [], "definitions": {"a": "b"}}, "bad_request", "content"),
    ({"content": [{"type": "text", "data": "x", "url": "u"}], "definitions": {"a": "b"}}, "bad_request", "content.0"),
    ({"content": [{"type": "text", "data": "x"}, {"type": "text", "data": "y"}], "definitions": {"a": "b"}}, "bad_request", "content.1.type"),
    ({"content": [{"type": "image", "data": "x"}], "definitions": {"a": "b"}}, "bad_request", "content.0.type"),
    ({"content": [{"type": "text", "data": "  "}], "definitions": {"a": "b"}}, "bad_request", "content.0.data"),
    (body({"a": "b"}, 0), "bad_request", "threshold"), (body({"a": "b"}, 1), "bad_request", "threshold"), (body({"a": "b"}, "0.5"), "bad_request", "threshold"),
    (body({"a": "b"}, True), "bad_request", "threshold"),
    ({"content": [{"type": "text", "data": "x"}]}, "bad_request", "definitions"),
    (body({}), "bad_request", "definitions"),
    (body({"Threat": "b"}), "bad_request", "definitions.Threat"), (body({"1a": "b"}), "bad_request", "definitions.1a"), (body({"a-b": "b"}), "bad_request", "definitions.a-b"),
    (body({"a\n": "b"}), "bad_request", "definitions.a\n"), (body({"a": ""}), "bad_request", "definitions.a"), (body({"a": 5}), "bad_request", "definitions.a"),
    (body({f"d{i}": "x" for i in range(MAX_DEFINITIONS + 1)}), "too_large", "definitions"),
    (body({"a": "x" * (MAX_DEFINITION_CHARS + 1)}), "too_large", "definitions.a"),
    (body({"a": "x"}, text="t" * (MAX_TEXT_CHARS + 1)), "too_large", "content.0"),
])
def test_refused_requests_use_the_error_object(b, code, path):
    stub = Stub({})
    with pytest.raises(KevError) as e: detect(stub, b)
    assert e.value.code == code and e.value.path == path and e.value.status == (413 if code == "too_large" else 422) and not stub.requests


def test_requests_at_the_limits_are_accepted():
    defs = {f"d{i}": "x" * MAX_DEFINITION_CHARS for i in range(MAX_DEFINITIONS)}
    r = detect(Stub({"x" * MAX_DEFINITION_CHARS: 0.1}), body(defs, 0.999999, text="t" * MAX_TEXT_CHARS))
    assert len(r["scores"]) == MAX_DEFINITIONS and r["flagged"] == []


def test_the_module_says_it_is_experimental_and_not_calibrated():
    import kev.detect
    assert "EXPERIMENTAL" in kev.detect.__doc__ and "NOT calibrated" in kev.detect.__doc__ and "NOT calibrated" in kev.detect.EXPERIMENTAL_NOTE
