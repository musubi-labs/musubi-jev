"""EXPERIMENTAL Detect-shaped adapter: open-vocabulary definitions scored by Kev. Not part of Detect.

    POST /v1/detect   (only when kev.serve runs with --demo)
    {"content": [{"type": "text", "data": "..."}], "definitions": {"threat": "The text states or implies harm to a person."}, "threshold": 0.5}
 -> {"model": "kev/v0.1.0/<16 hex>", "scores": {"threat": 0.012345}, "flagged": []}

Each definition becomes one yes/no (Noul) question asked of the content as the state. The score is the probability of "yes",
rounded down to 6 decimals. A definition is flagged when its score is at or above the threshold, strongest first, ties by id.
The wire follows the Detect contract (docs/contract/README.md in image-label-fast): the response fields, the 6-decimal rule, the
`model` string and the error object.

These scores are NOT calibrated for this use. Kev was trained on public classification and QA data with roughly balanced classes,
and its temperature was fitted on those sets, so 0.5 does not mean what it means for a Detect category, and a clean text can score
well above 0. Kev has not been measured on moderation definitions at low prevalence. This adapter stays out of Detect until a
frozen definition suite gates it. The text is also read beyond the 384 state tokens the model was trained on.
"""
import hashlib, json, re
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from .api import SystemOneRequest
from .errors import KevError

NAME, RELEASE = "kev", "v0.1.0"      # bump RELEASE when the mapping below changes
QUESTION = "Does the content meet this definition? "
MAX_DEFINITIONS, MAX_DEFINITION_CHARS, MAX_TEXT_CHARS = 32, 2_000, 100_000
ID = re.compile(r"[a-z][a-z0-9_]*")
GRID = Decimal("0.000001")
EXPERIMENTAL_NOTE = ("Experimental. Scores each definition as a yes/no Kev question about the text. NOT calibrated: 0.5 does not yet mean what it "
                     "means for Detect categories, and this route stays out of Detect until a frozen definition suite gates it.")


def _bad(message, path=None): return KevError("bad_request", message, path)


def floor6(p: float) -> float:
    return float(Decimal(repr(min(max(float(p), 0.0), 1.0))).quantize(GRID, rounding=ROUND_FLOOR))


def parse(body) -> tuple[str, dict[str, str], float]:
    """Validate a request body in the contract's order: shape, parts, threshold, definitions, then sizes. -> (text, definitions, threshold)."""
    if not isinstance(body, dict): raise _bad("the request must be a JSON object")
    for k in body:
        if k not in ("content", "definitions", "threshold"): raise _bad(f"unknown field {k!r}", k)
    content = body.get("content")
    if not isinstance(content, list) or not content: raise _bad("content must be a non-empty list of parts", "content")
    text, text_at, seen = None, None, set()
    for i, part in enumerate(content):
        if not isinstance(part, dict) or set(part) != {"type", "data"}: raise _bad("a part is {type, data}", f"content.{i}")
        t = part["type"]
        if not isinstance(t, str) or not t: raise _bad("type must be a string", f"content.{i}.type")
        if t in seen: raise _bad(f"two parts of type {t!r}", f"content.{i}.type")
        seen.add(t)
        if t != "text": raise _bad(f"no definition reads a part of type {t!r}; only text is read", f"content.{i}.type")
        if not isinstance(part["data"], str) or not part["data"].strip(): raise _bad("text data must be a non-empty string", f"content.{i}.data")
        text, text_at = part["data"], i
    threshold = body.get("threshold", 0.5)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise _bad("threshold must be a number above 0 and below 1", "threshold")
    defs = body.get("definitions")
    if not isinstance(defs, dict) or not defs: raise _bad("definitions must be a non-empty object of id to definition", "definitions")
    for k, v in defs.items():
        if not ID.fullmatch(k): raise _bad(f"definition id {k!r} must match [a-z][a-z0-9_]*", f"definitions.{k}")
        if not isinstance(v, str) or not v.strip(): raise _bad("a definition must be a non-empty string", f"definitions.{k}")
    if len(defs) > MAX_DEFINITIONS: raise KevError("too_large", f"at most {MAX_DEFINITIONS} definitions, got {len(defs)}", "definitions")
    for k, v in defs.items():
        if len(v) > MAX_DEFINITION_CHARS: raise KevError("too_large", f"a definition is {len(v)} characters; the limit is {MAX_DEFINITION_CHARS}", f"definitions.{k}")
    if len(text) > MAX_TEXT_CHARS: raise KevError("too_large", f"text is {len(text)} characters; the limit is {MAX_TEXT_CHARS}", f"content.{text_at}")
    return text, defs, float(threshold)


def model_string(identity: str, definitions: dict[str, str]) -> str:
    """kev/<release>/<16 hex>: sha256 over the checkpoint identity, the question template and the definitions sorted by id."""
    blob = json.dumps({"checkpoint": identity, "template": QUESTION, "definitions": sorted(definitions.items())}, ensure_ascii=False, separators=(",", ":"))
    return f"{NAME}/{RELEASE}/{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def flag(scores: dict[str, float], threshold: float) -> list[str]:
    """Ids at or above the threshold, strongest first, ties by id. A threshold finer than 6 decimals counts as the next 6-decimal number up."""
    line = float(Decimal(repr(threshold)).quantize(GRID, rounding=ROUND_CEILING))
    return sorted((k for k, s in scores.items() if s >= line), key=lambda k: (-scores[k], k))


def detect(kev, body) -> dict:
    """`kev` is a kev.runtime.Kev (or anything with answer(request) and identity())."""
    text, definitions, threshold = parse(body)
    ids = sorted(definitions)
    req = SystemOneRequest.model_validate({"state": text, "questions": {i: {"type": "noul", "instructions": QUESTION + definitions[i]} for i in ids}})
    res = kev.answer(req)
    scores = {i: floor6(p[1]) for i, p in zip(ids, res.probabilities)}
    return {"model": model_string(kev.identity(), definitions), "scores": scores, "flagged": flag(scores, threshold)}
