"""kev.runtime.Kev on a tiny random checkpoint (tests/conftest.py): loading, deciding, limits, error shape, the lock, the prefix cache."""
import dataclasses, json, threading, time

import pytest
import torch

from kev import limits
from kev.api import SystemOneRequest, to_answers, to_record
from kev.checkpoint import HeadMeta
from kev.config import Config
from kev.errors import KevError
from kev.runtime import Kev, Result

STATE = "I was charged twice. Please fix this ASAP."
QS = {"billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
      "tone": {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": None, "frustrated": None, "angry": None}},
      "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["can wait", "this week", "today"]}}


def test_from_pretrained_and_decide_match_the_direct_model_path(tiny_kev):
    answers = tiny_kev.decide(STATE, QS)
    assert set(answers) == set(QS) and answers["billing"]["type"] == "noul" and answers["tone"]["choice"] in {"calm", "frustrated", "angry"}
    assert set(answers["urgency"]["legend"]) == {"0", "1", "2"}
    rec, meta = to_record(SystemOneRequest.model_validate({"state": STATE, "questions": QS}))
    m = tiny_kev.model
    ps = m.probs(m.encode(tiny_kev.tok, rec, strict=True))
    assert answers == to_answers([p.tolist() for p in ps], meta)
    assert tiny_kev.decide(STATE, QS) == answers


def test_answer_returns_raw_probabilities_and_usage(tiny_kev):
    r = tiny_kev.answer(SystemOneRequest.model_validate({"state": STATE, "questions": QS}))
    assert isinstance(r, Result) and [len(p) for p in r.probabilities] == [2, 3, 3] and all(abs(sum(p) - 1) < 1e-5 for p in r.probabilities)
    assert r.usage["input_tokens"] > r.state_tokens > 0 and r.usage["output_tokens"] > 0


def test_checkpoint_metadata_and_calibration(tiny_kev, tiny_checkpoint):
    meta = tiny_kev.meta
    assert isinstance(meta, HeadMeta) and meta.head_dim == 8 and meta.lora == 4 and meta.temperature == 2.0
    assert tiny_kev.model.head.temperature == 2.0
    assert meta.calibration() == {"calibrated": True, "method": "temperature", "temperature": 2.0, "source": "checkpoint"}
    raw = Kev.from_pretrained(tiny_checkpoint, device="cpu", config=Config(temperature=1.0))
    assert raw.meta.calibration() == {"calibrated": False, "method": "temperature", "temperature": 1.0, "source": "override"} and raw.model.head.temperature == 1.0
    p_cal = tiny_kev.answer(SystemOneRequest.model_validate({"state": STATE, "questions": QS})).probabilities
    p_raw = raw.answer(SystemOneRequest.model_validate({"state": STATE, "questions": QS})).probabilities
    assert [max(range(len(a)), key=a.__getitem__) for a in p_cal] == [max(range(len(a)), key=a.__getitem__) for a in p_raw]
    assert p_cal != p_raw


def test_head_meta_without_a_temperature_is_uncalibrated():
    meta = HeadMeta.from_dict({"base": "b", "head": {}})
    assert not meta.calibrated and meta.applied_temperature == 1.0 and meta.calibration()["source"] == "none"
    assert meta.head_dim == 256 and not meta.option_isolation


def test_from_pretrained_errors_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"): Kev.from_pretrained(str(tmp_path / "nope"), device="cpu")
    with pytest.raises(FileNotFoundError, match="head.pt"): Kev.from_pretrained(str(tmp_path), device="cpu")


def test_config_reads_every_variable_once_and_rejects_bad_values():
    c = Config.from_env({"KEV_DTYPE": "bf16", "KEV_MERGE": "0", "KEV_ATTN": "sdpa", "KEV_SHAPE_BUCKET": "1", "KEV_PREFIX_CACHE": "0", "KEV_PREFIX_MIN_TOKENS": "10",
                         "KEV_DATE_FACTS": "1", "KEV_TEMPERATURE": "1.0", "KEV_LORA_SCALE": "0.5", "KEV_DEMO": "1", "KEV_QUEUE_TIMEOUT": "3"})
    assert c == Config(dtype="bf16", merge=False, attn="sdpa", shape_bucket=1, prefix_cache=0, prefix_min_tokens=10, date_facts=True, temperature=1.0,
                       lora_scale=0.5, demo=True, queue_timeout=3.0)
    assert Config.from_env({}) == Config() and Config().merge and Config().prefix_cache == 4 and Config().prefix_min_tokens == 384
    assert {f.metadata["env"] for f in dataclasses.fields(Config)} == {"KEV_DTYPE", "KEV_MERGE", "KEV_ATTN", "KEV_SHAPE_BUCKET", "KEV_PREFIX_CACHE", "KEV_PREFIX_MIN_TOKENS",
                                                                      "KEV_DATE_FACTS", "KEV_TEMPERATURE", "KEV_LORA_SCALE", "KEV_DEMO", "KEV_QUEUE_TIMEOUT"}
    assert all(f.metadata["doc"] for f in dataclasses.fields(Config))
    with pytest.raises(ValueError): Config.from_env({"KEV_DTYPE": "int4"})
    with pytest.raises(dataclasses.FrozenInstanceError): Config().dtype = "bf16"


def noul(n):
    return {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(n)}


@pytest.mark.parametrize("state,questions,path", [
    ("s", noul(limits.MAX_QUESTIONS + 1), "questions"),
    ("s", {f"q{i}": {"type": "choice", "instructions": "x", "criteria": {f"o{j}": None for j in range(limits.MAX_OPTIONS)}} for i in range(3)}, "questions"),
    ("s" * (limits.MAX_STATE_CHARS + 1), noul(1), "state"),
    ("s", {"big": {"type": "noul", "instructions": "x" * (limits.MAX_INSTRUCTION_CHARS + 1)}}, "questions.big.instructions"),
    ("s", {"big": {"type": "choice", "instructions": "x", "criteria": {"a": "y" * limits.MAX_OPTION_CHARS, "b": None}}}, "questions.big.criteria"),
    ("s", {"big": {"type": "noul", "instructions": "x", "criteria": {"true": "y" * limits.MAX_OPTION_CHARS}}}, "questions.big.criteria"),
])
def test_limits_are_refused_as_too_large_before_the_model_runs(tiny_kev, state, questions, path):
    with pytest.raises(KevError) as e: tiny_kev.decide(state, questions)
    assert e.value.code == "too_large" and e.value.status == 413 and e.value.path == path and not e.value.retryable


def test_a_state_over_the_token_cap_is_refused_not_truncated(tiny_kev):
    n = limits.MAX_STATE_TOKENS + 50
    assert len(tiny_kev.tok("a" * n, add_special_tokens=False).input_ids) > limits.MAX_STATE_TOKENS
    with pytest.raises(KevError) as e: tiny_kev.decide("a" * n, noul(1))
    assert e.value.code == "too_large" and e.value.path == "state"


def test_a_branch_over_the_token_cap_names_its_question(tiny_kev):
    with pytest.raises(KevError) as e: tiny_kev.decide("s", {"ok": {"type": "noul", "instructions": "x"}, "long": {"type": "noul", "instructions": "a" * (limits.MAX_INSTRUCTION_CHARS - 1)}})
    assert e.value.code == "too_large" and e.value.path == "questions.long"


@pytest.mark.parametrize("questions,path", [
    ({"q": {"type": "bogus", "instructions": "i"}}, "questions.q"),
    ({"q": {"type": "score", "instructions": "i", "criteria": ["only one"]}}, "questions.q.criteria"),
    ({}, "questions"),
    ({"q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}}, "questions.q.criteria"),
])
def test_invalid_requests_are_bad_request_with_a_path(tiny_kev, questions, path):
    with pytest.raises(KevError) as e: tiny_kev.decide("s", questions)
    assert e.value.code == "bad_request" and e.value.status == 422 and e.value.path == path


def test_error_object_shape():
    assert KevError("too_large", "m", "state").body() == {"error": {"code": "too_large", "message": "m", "path": "state", "retryable": False}}
    assert KevError("unavailable", "m").body() == {"error": {"code": "unavailable", "message": "m", "retryable": True}}
    assert {c: KevError(c, "m").status for c in ("bad_request", "too_large", "model_error", "unavailable")} == {"bad_request": 422, "too_large": 413, "model_error": 500, "unavailable": 503}
    with pytest.raises(ValueError): KevError("nope", "m")


def test_a_model_failure_is_model_error_without_internals(tiny_kev, monkeypatch):
    def boom(rec): raise RuntimeError("secret path /x/y")
    monkeypatch.setattr(tiny_kev, "_probs", boom)
    with pytest.raises(KevError) as e: tiny_kev.decide("s", noul(1))
    assert e.value.code == "model_error" and "secret" not in e.value.message and not tiny_kev.lock.locked()


def test_a_busy_model_is_unavailable_and_retryable(tiny_kev):
    k = Kev(tiny_kev.tok, tiny_kev.model, "cpu", Config(queue_timeout=0.05))
    with k.lock:
        with pytest.raises(KevError) as e: k.decide("s", noul(1))
    assert e.value.code == "unavailable" and e.value.retryable and e.value.status == 503


def test_prefix_cache_hit_gives_the_same_probabilities(tiny_checkpoint):
    cached = Kev.from_pretrained(tiny_checkpoint, device="cpu", config=Config(prefix_min_tokens=0, prefix_cache=2))
    plain = Kev.from_pretrained(tiny_checkpoint, device="cpu", config=Config(prefix_cache=0))
    req = SystemOneRequest.model_validate({"state": STATE * 3, "questions": QS})
    a, b, c = cached.answer(req), cached.answer(req), plain.answer(req)
    assert (a.prefix_cache_hit, b.prefix_cache_hit, c.prefix_cache_hit) == (False, True, False) and (cached.prefix_misses, cached.prefix_hits) == (1, 1)
    for x, y, z in zip(a.probabilities, b.probabilities, c.probabilities):
        assert max(abs(i - j) for i, j in zip(x, y)) < 1e-5 and max(abs(i - j) for i, j in zip(x, z)) < 1e-5
    for i in range(4): cached.decide(f"state {i} " * 4, noul(1))
    assert len(cached.prefix_cache) == 2


def test_date_facts_setting_changes_the_state_the_model_reads(tiny_checkpoint):
    on = Kev.from_pretrained(tiny_checkpoint, device="cpu", config=Config(date_facts=True))
    off = Kev.from_pretrained(tiny_checkpoint, device="cpu")
    req = SystemOneRequest.model_validate({"state": "Due July 4, 2026. Received June 26, 2026.", "questions": noul(1)})
    assert on.answer(req).input_tokens > off.answer(req).input_tokens


def test_concurrent_decisions_do_not_interleave_and_match_serial(tiny_kev):
    states = [f"ticket {i}: " + STATE for i in range(4)]
    serial = [tiny_kev.decide(s, QS) for s in states]
    out, errors = {}, []
    def work(i):
        try: out[i] = [tiny_kev.decide(states[i % 4], QS) for _ in range(5)]
        except Exception as e: errors.append(e)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors and all(out[i] == [serial[i % 4]] * 5 for i in range(8))


class Tok:
    def __call__(self, text, add_special_tokens=False): return type("E", (), {"input_ids": list(range(len(text)))})()


class SlowModel:
    """A model double that records how many calls are inside it at once."""
    meta = HeadMeta("b", None, 8, False, None, 4, None)

    def __init__(self): self.inside = self.peak = 0; self.guard = threading.Lock()

    def encode(self, tok, rec, **kw): return {"ids": [1, 2, 3, 4], "seg": [0, 0, 1, 1], "pos": [0, 1, 2, 3]}

    def probs(self, enc):
        with self.guard: self.inside += 1; self.peak = max(self.peak, self.inside)
        time.sleep(0.02)
        with self.guard: self.inside -= 1
        return [torch.tensor([0.25, 0.75])]


def test_the_lock_serialises_threads_over_the_shared_model():
    model = SlowModel(); k = Kev(Tok(), model, "cpu", Config(prefix_cache=0))
    threads = [threading.Thread(target=k.decide, args=("s", noul(1))) for _ in range(6)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert model.peak == 1
    assert k.decide("s", noul(1)) == {"q0": {"type": "noul", "noul": 0.75}}


def test_relative_paths_are_never_taken_for_hub_ids():
    for p in ("runs/nope", "./runs/nope", "runs/a/b"):
        with pytest.raises(FileNotFoundError, match="not found"): Kev.from_pretrained(p, device="cpu")


def test_only_control_token_and_length_errors_become_bad_request(tiny_kev, monkeypatch):
    from kev.errors import KevError
    def boom(rec): raise ValueError("internal detail")
    monkeypatch.setattr(tiny_kev, "_probs", boom)
    with pytest.raises(KevError) as e: tiny_kev.decide("s", {"q": {"type": "noul", "instructions": "x"}})
    assert e.value.code == "model_error" and "internal" not in e.value.message
