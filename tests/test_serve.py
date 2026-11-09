"""kev.serve on a tiny random checkpoint (tests/conftest.py), through FastAPI's TestClient: no weights, no network."""
import re, threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from kev import limits
from kev.config import Config
from kev.runtime import Kev
from kev.serve import create_app

STATE = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
DEPARTMENT = {"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}
REQ = {"state": STATE, "model": "kev-latest", "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT},
    "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}}}


@pytest.fixture(scope="module")
def client(tiny_kev):
    return TestClient(create_app(tiny_kev), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def demo_client(tiny_kev):
    return TestClient(create_app(tiny_kev, demo=True), raise_server_exceptions=False)


def assert_error(r, code, status, path=None):
    assert r.status_code == status and set(r.json()) == {"error"}, r.text
    e = r.json()["error"]
    assert e["code"] == code and isinstance(e["message"], str) and e["retryable"] is (code == "unavailable")
    assert e.get("path") == path and set(e) <= {"code", "message", "path", "retryable"}


def test_systemone_wire_shape(client):
    r = client.post("/v1/systemone", json=REQ)
    assert r.status_code == 200
    body = r.json()
    assert list(body) == ["model", "answers", "usage", "latency_ms"] and body["model"] == "kev-latest" and set(body["usage"]) == {"input_tokens", "output_tokens"}
    a = body["answers"]
    assert set(a["department"]) == {"type", "choice", "confidence", "probabilities"} and set(a["department"]["probabilities"]) == set(DEPARTMENT)
    assert set(a["escalate"]) == {"type", "noul"} and 0 <= a["escalate"]["noul"] <= 1
    assert set(a["frustration"]) == {"type", "score", "legend", "probabilities", "confidence"} and set(a["frustration"]["probabilities"]) == {"0", "1", "2"}


def test_systemone_echoes_any_model_name_and_matches_the_python_api(client, tiny_kev):
    r = client.post("/v1/systemone", json={**REQ, "model": "jev-latest"}).json()
    assert r["model"] == "jev-latest" and r["answers"] == tiny_kev.decide(STATE, REQ["questions"])


def test_models_keeps_its_fields_and_reports_calibration(client, tiny_checkpoint):
    m = client.get("/v1/models").json()["models"][0]
    assert {"id", "aliases", "run", "base", "temperature"} <= set(m) and m["id"] == "kev-latest" and m["aliases"] == ["jev-latest"] and m["run"] == tiny_checkpoint
    assert m["temperature"] == 2.0 and m["calibration"] == {"calibrated": True, "method": "temperature", "temperature": 2.0, "source": "checkpoint"}
    assert m["config"]["dtype"] == "fp32" and m["limits"]["max_questions"] == limits.MAX_QUESTIONS


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize("questions,path", [
    ({"q": {"type": "score", "instructions": "i", "criteria": ["only one"]}}, "questions.q.criteria"),
    ({"q": {"type": "bogus", "instructions": "i"}}, "questions.q"),
    ({}, "questions"),
    ({"q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}}, "questions.q.criteria"),
])
def test_validation_errors_are_422_in_the_one_shape(client, questions, path):
    assert_error(client.post("/v1/systemone", json={"state": "x", "model": "m", "questions": questions}), "bad_request", 422, path)


def test_malformed_json_and_missing_fields_are_bad_request(client):
    assert_error(client.post("/v1/systemone", content=b"{nope", headers={"content-type": "application/json"}), "bad_request", 422, "body")
    assert client.post("/v1/systemone", json={"state": "x"}).status_code == 422


def test_limits_are_413_too_large(client):
    qs = {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(limits.MAX_QUESTIONS + 1)}
    assert_error(client.post("/v1/systemone", json={"state": "s", "questions": qs}), "too_large", 413, "questions")
    assert_error(client.post("/v1/systemone", json={"state": "s" * (limits.MAX_STATE_CHARS + 1), "questions": {"q": REQ["questions"]["escalate"]}}), "too_large", 413, "state")
    assert_error(client.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "noul", "instructions": "x" * (limits.MAX_INSTRUCTION_CHARS + 1)}}}),
                 "too_large", 413, "questions.q.instructions")


def test_a_body_over_the_cap_is_refused_before_it_is_read(client):
    r = client.post("/v1/systemone", content=b" " * (limits.MAX_BODY_BYTES + 1), headers={"content-type": "application/json"})
    assert_error(r, "too_large", 413, "body")


def test_a_model_failure_is_500_model_error(tiny_kev, monkeypatch):
    c = TestClient(create_app(tiny_kev), raise_server_exceptions=False)
    monkeypatch.setattr(tiny_kev, "_probs", lambda rec: 1 / 0)
    r = c.post("/v1/systemone", json=REQ)
    assert_error(r, "model_error", 500) and "division" not in r.text


def test_a_busy_model_is_503_unavailable_and_retryable(tiny_kev):
    k = Kev(tiny_kev.tok, tiny_kev.model, "cpu", Config(queue_timeout=0.05))
    c = TestClient(create_app(k), raise_server_exceptions=False)
    with k.lock:
        assert_error(c.post("/v1/systemone", json=REQ), "unavailable", 503)


def test_unknown_routes_use_the_error_shape(client):
    assert_error(client.get("/nowhere"), "bad_request", 404)
    assert_error(client.get("/v1/systemone"), "bad_request", 405)


DEMO_ROUTES = [("/v1/systemone/permute", {"request": REQ, "question": "department", "n_perm": 2}), ("/v1/systemone/separate", REQ),
               ("/v1/detect", {"content": [{"type": "text", "data": "x"}], "definitions": {"a": "b"}}), ("/api/info", None), ("/api/predict", {"state": "s", "questions": []})]


@pytest.mark.parametrize("path,body", DEMO_ROUTES)
def test_demo_and_legacy_routes_are_off_by_default(client, path, body):
    r = client.post(path, json=body) if body is not None else client.get(path)
    assert r.status_code in (404, 405), r.text


def test_demo_switch_turns_on_permute_separate_and_detect(demo_client):
    r = demo_client.post("/v1/systemone/permute", json={"request": REQ, "question": "department", "n_perm": 3})
    assert r.status_code == 200 and len(r.json()["runs"]) == 3 and set(r.json()["spread"]) == set(DEPARTMENT)
    r = demo_client.post("/v1/systemone/separate", json=REQ)
    assert r.status_code == 200 and set(r.json()["answers"]) == set(REQ["questions"]) and r.json()["usage"]["input_tokens"] > 0
    assert_error(demo_client.post("/v1/systemone/permute", json={"request": REQ, "question": "escalate"}), "bad_request", 422, "question")
    assert_error(demo_client.post("/v1/systemone/permute", json={"request": REQ, "question": "department", "n_perm": 999}), "bad_request", 422, "n_perm")
    r = demo_client.post("/v1/detect", json={"content": [{"type": "text", "data": STATE}], "definitions": {"late": "The order arrived late.", "billing": "A charge is disputed."}})
    assert r.status_code == 200 and set(r.json()) == {"model", "scores", "flagged"} and re.fullmatch(r"kev/v\d+\.\d+\.\d+/[0-9a-f]{16}", r.json()["model"])
    assert_error(demo_client.post("/v1/detect", json={"content": [], "definitions": {"a": "b"}}), "bad_request", 422, "content")
    assert demo_client.get("/v1/models").status_code == 200 and "/v1/detect" in demo_client.get("/openapi.json").json()["paths"]


def test_separate_matches_packed_within_noise(demo_client):
    packed = demo_client.post("/v1/systemone", json=REQ).json()["answers"]; sep = demo_client.post("/v1/systemone/separate", json=REQ).json()["answers"]
    for k in DEPARTMENT: assert abs(packed["department"]["probabilities"][k] - sep["department"]["probabilities"][k]) <= 0.011


def test_parallel_requests_get_the_same_answers_as_serial(client):
    want = client.post("/v1/systemone", json=REQ).json()["answers"]
    out, errors = [], []
    def work():
        try: out.append(client.post("/v1/systemone", json=REQ).json()["answers"])
        except Exception as e: errors.append(e)
    threads = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors and out == [want] * 6
