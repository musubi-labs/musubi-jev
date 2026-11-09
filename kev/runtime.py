"""Kev in a process: load a checkpoint once, answer typed questions.

    from kev import Kev
    kev = Kev.from_pretrained("runs/kev")          # or a Hub id such as jaredpalmer/kev-4b
    answers = kev.decide("I was charged twice. Please fix this ASAP.",
                         {"billing": {"type": "noul", "instructions": "Is this ticket about billing?"}})

`decide` returns the `answers` object of POST /v1/systemone. kev.serve is a thin HTTP layer over `answer`, which also
returns the raw probabilities and the usage figures. Calls are serialised by one lock, because the tokenizer, the model and
the prefix cache are shared; a caller that waits longer than Config.queue_timeout gets the error `unavailable`.
Every failure is a kev.errors.KevError.
"""
import logging, os, threading, time
from dataclasses import dataclass, replace

import torch
from pydantic import ValidationError

from . import limits
from .api import SystemOneRequest, output_tokens, to_answers, to_record, with_date_facts
from .config import Config
from .errors import KevError
from .evaluate import load, resolve_run
from .model import ControlToken, TooLong

log = logging.getLogger("kev")
_UNION_TAGS = {"noul", "choice", "score"}


@dataclass
class Result:
    answers: dict                       # the `answers` object of /v1/systemone
    probabilities: list[list[float]]    # unrounded, one list per question in request order
    input_tokens: int
    output_tokens: int
    state_tokens: int
    latency_ms: float                   # model time, without queueing
    prefix_cache_hit: bool

    @property
    def usage(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def validation_error(errors: list[dict]) -> KevError:
    """KevError (bad_request) for the first pydantic error. The path drops FastAPI's leading `body` and the question-type tag
    that pydantic adds for a discriminated union (questions.<id>.<type>.criteria -> questions.<id>.criteria)."""
    first = errors[0] if errors else {"loc": (), "msg": "invalid request"}
    loc = ["body"] if first.get("type") == "json_invalid" else [str(x) for x in first["loc"] if x != "body"]
    if len(loc) > 2 and loc[0] == "questions" and loc[2] in _UNION_TAGS: del loc[2]
    return KevError("bad_request", first["msg"], ".".join(loc) or None)


def check_counts(req: SystemOneRequest):
    if len(req.questions) > limits.MAX_QUESTIONS:
        raise KevError("too_large", f"at most {limits.MAX_QUESTIONS} questions per request, got {len(req.questions)}", "questions")
    total = 0
    for qid, q in req.questions.items():
        n = 2 if q.type == "noul" else len(q.criteria)
        total += n
        if n > limits.MAX_OPTIONS: raise KevError("too_large", f"at most {limits.MAX_OPTIONS} options per question, got {n}", f"questions.{qid}.criteria")
    if total > limits.MAX_TOTAL_OPTIONS: raise KevError("too_large", f"at most {limits.MAX_TOTAL_OPTIONS} options per request, got {total}", "questions")


def check_text(rec: dict, meta: list[dict]):
    if len(rec["state"]) > limits.MAX_STATE_CHARS:
        raise KevError("too_large", f"state is {len(rec['state'])} characters; the limit is {limits.MAX_STATE_CHARS}", "state")
    for q, m in zip(rec["questions"], meta):
        if len(q["instr"]) > limits.MAX_INSTRUCTION_CHARS:
            raise KevError("too_large", f"instructions are {len(q['instr'])} characters; the limit is {limits.MAX_INSTRUCTION_CHARS}", f"questions.{m['id']}.instructions")
        for o in q["options"]:
            if len(o) > limits.MAX_OPTION_CHARS:
                raise KevError("too_large", f"an option is {len(o)} characters; the limit is {limits.MAX_OPTION_CHARS}", f"questions.{m['id']}.criteria")


def _sync(dev):
    if dev == "mps": torch.mps.synchronize()
    elif dev == "cuda": torch.cuda.synchronize()


def pick_device(device: str = "auto") -> str:
    if device != "auto": return device
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Kev:
    def __init__(self, tok, model, device: str, config: Config | None = None, path: str | None = None, label: str | None = None):
        self.tok, self.model, self.device = tok, model, device
        self.config = config or Config()
        self.path, self.label = path, label or path
        self.lock = threading.Lock()
        self.prefix_cache, self.prefix_hits, self.prefix_misses = {}, 0, 0

    @classmethod
    def from_pretrained(cls, run: str, device: str = "auto", config: Config | None = None, label: str | None = None) -> "Kev":
        """`run`: a checkpoint directory (adapter_model.safetensors, adapter_config.json, head.pt) or a Hub id, optionally
        pinned with `@revision`. Settings come from `config` (default: the KEV_* environment)."""
        config = config or Config.from_env()
        dev = pick_device(device)
        if dev == "mps" and config.attn is None: config = replace(config, attn="sdpa")   # serving default on Apple GPUs (parity measured)
        path = resolve_run(run)
        if not os.path.isfile(f"{path}/head.pt"): raise FileNotFoundError(f"{path} is not a Kev checkpoint: head.pt is missing")
        tok, model = load(path, dev, config=config)
        return cls(tok, model, dev, config, path=path, label=label or run)

    @property
    def meta(self):
        return self.model.meta

    def info(self) -> dict:
        """What GET /v1/models reports about the loaded checkpoint."""
        m = self.meta
        return {"run": self.label, "base": m.base, "temperature": m.applied_temperature, "calibration": m.calibration(),
                "config": self.config.describe(), "limits": {k.lower(): v for k, v in vars(limits).items() if k.isupper()}}


    def decide(self, state, questions) -> dict:
        """Typed questions about one state -> the `answers` object of /v1/systemone. `questions` maps an id to a Noul, Choice
        or Score question in the API shape (dicts or the kev.api models)."""
        try: req = SystemOneRequest.model_validate({"state": state, "questions": questions})
        except ValidationError as e: raise validation_error(e.errors()) from None
        return self.answer(req).answers

    def answer(self, req: SystemOneRequest) -> Result:
        check_counts(req)
        if self.config.date_facts: req = req.model_copy(update={"state": with_date_facts(req.state)})
        rec, meta = to_record(req)
        check_text(rec, meta)
        if not self.lock.acquire(timeout=self.config.queue_timeout):
            raise KevError("unavailable", "the model is busy; retry")
        try:
            try:
                ps, info = self._probs(rec)
                answers = to_answers(ps, meta)
                return Result(answers, ps, info["tokens"], output_tokens(self.tok, answers), info["state_tokens"], info["latency_ms"], info["prefix_cache_hit"])
            except TooLong as e:
                path = "state" if e.part == "state" else f"questions.{meta[e.index - 1]['id']}"
                raise KevError("too_large", str(e), path) from None
            except KevError: raise
            except ControlToken as e: raise KevError("bad_request", str(e)) from None
            except Exception:
                log.exception("inference failed")
                raise KevError("model_error", "the model failed on this request") from None
        finally:
            self.lock.release()

    def _probs(self, rec):
        """One forward pass; the state prefix (tokens up to the first question) is cached across requests, so a repeated state
        only pays for its question branches. Exactness: the state's activations do not depend on the branches. Call with the lock held."""
        model, cfg, cache = self.model, self.config, self.prefix_cache
        enc = model.encode(self.tok, rec, max_state=limits.MAX_STATE_TOKENS, max_branch=limits.MAX_BRANCH_TOKENS, strict=True)
        Ls = enc["seg"].count(0); key = (tuple(enc["ids"][:Ls]), bool(enc.get("option_isolation")))
        _sync(self.device); t = time.time()
        eligible = cfg.prefix_cache and Ls >= cfg.prefix_min_tokens
        if eligible and key in cache:
            prefix = cache.pop(key)                       # pop + reinsert = LRU order
            ps = model.probs_with_prefix(enc, prefix); cache[key] = prefix
            self.prefix_hits += 1; hit = True
        elif eligible:
            ps, prefix = model.probs_and_prefix(enc)      # one pass, and the state prefix is kept for next time
            cache[key] = prefix
            while len(cache) > cfg.prefix_cache: cache.pop(next(iter(cache)))
            self.prefix_misses += 1; hit = False
        else:
            ps = model.probs(enc); hit = False
        _sync(self.device); dt = time.time() - t
        return [p.tolist() for p in ps], {"tokens": len(enc["ids"]), "state_tokens": Ls, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": hit}
