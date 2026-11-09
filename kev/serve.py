"""FastAPI server over one loaded checkpoint: prefill-only decisions, TypeSafe-compatible.

Run: uv run --extra serve python -m kev.serve --run runs/kev --port 8008

Production surface: POST /v1/systemone, GET /v1/models, GET /healthz. `--demo` (or KEV_DEMO=1) adds the routes the playground
and the experiments use: POST /v1/systemone/permute, POST /v1/systemone/separate and the experimental POST /v1/detect. Every
failure is {"error": {"code", "message", "path"?, "retryable"}} (kev.errors). The body limit checks Content-Length; the limits in
kev.limits apply to what a request says after it is parsed.
"""
import argparse, os, random, re

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import limits
from .api import SystemOneRequest
from .detect import EXPERIMENTAL_NOTE, detect
from .errors import KevError
from .runtime import Kev, validation_error


class PermuteSystemOne(BaseModel):
    request: SystemOneRequest
    question: str
    n_perm: int = Field(default=6, ge=1, le=24)
    seed: int = 0


def _error(err: KevError) -> JSONResponse:
    return JSONResponse(err.body(), status_code=err.status)


def create_app(kev: Kev, demo: bool = False) -> FastAPI:
    app = FastAPI(title="kev")

    @app.exception_handler(KevError)
    async def _kev_error(request: Request, exc: KevError):
        return _error(exc)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError):
        return _error(validation_error(exc.errors()))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = "bad_request" if exc.status_code < 500 else "model_error"   # unknown routes and methods keep their 404 and 405
        return JSONResponse({"error": {"code": code, "message": str(exc.detail), "retryable": False}}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        return _error(KevError("model_error", "internal error"))

    @app.middleware("http")
    async def _body_limit(request: Request, call_next):
        n = request.headers.get("content-length", "")
        if n.isdigit() and int(n) > limits.MAX_BODY_BYTES:
            return _error(KevError("too_large", f"request body is {n} bytes; the limit is {limits.MAX_BODY_BYTES}", "body"))
        return await call_next(request)

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        """TypeSafe-compatible endpoint: typed questions in, typed answers out, one prefill pass."""
        r = kev.answer(req)
        return {"model": req.model, "answers": r.answers, "usage": r.usage, "latency_ms": r.latency_ms}

    @app.get("/v1/models")
    def models():
        i = kev.info()
        return {"models": [{"id": "kev-latest", "aliases": ["jev-latest"], "run": i["run"], "base": i["base"], "temperature": i["temperature"],
                            "calibration": i["calibration"], "config": i["config"], "limits": i["limits"]}]}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    if demo:
        @app.post("/v1/systemone/permute")
        def systemone_permute(r: PermuteSystemOne):
            """Re-run one Choice question under n_perm option orders. Returns per-order probabilities keyed by option name."""
            q = r.request.questions.get(r.question)
            if q is None or q.type != "choice": raise KevError("bad_request", "question must be an existing choice question", "question")
            rng = random.Random(r.seed); keys = list(q.criteria); runs = []
            for i in range(r.n_perm):
                order = list(keys)
                if i > 0: rng.shuffle(order)
                req = r.request.model_copy(update={"questions": {r.question: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
                res = kev.answer(req); a = res.answers[r.question]
                runs.append({"order": order, "probabilities": a["probabilities"], "choice": a["choice"], "latency_ms": res.latency_ms})
            spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
            return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}

        @app.post("/v1/systemone/separate")
        def systemone_separate(req: SystemOneRequest):
            """Answer each question in its own request against the same state (N passes). For packed-vs-separate comparison."""
            answers, tokens, out_tokens, ms = {}, 0, 0, 0.0
            for qid, q in req.questions.items():
                res = kev.answer(req.model_copy(update={"questions": {qid: q}}))
                answers.update(res.answers); tokens += res.input_tokens; out_tokens += res.output_tokens; ms += res.latency_ms
            return {"model": req.model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": out_tokens}, "latency_ms": round(ms, 1)}

        @app.post("/v1/detect", description=EXPERIMENTAL_NOTE)
        def detect_route(body: dict = Body(...)):
            return detect(kev, body)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/kev")
    ap.add_argument("--fallback", default="runs/smoke")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--demo", action="store_true", help="also serve /v1/systemone/permute, /v1/systemone/separate and the experimental /v1/detect")
    a = ap.parse_args()
    from .config import Config
    config = Config.from_env()
    is_hub_id = re.fullmatch(r"[\w.-]+/[\w.-]+", a.run) and not os.path.isdir(a.run)
    run = a.run if is_hub_id or os.path.exists(f"{a.run}/head.pt") else a.fallback
    if run != a.run: print(f"{a.run} not found, falling back to {run}")
    kev = Kev.from_pretrained(run, config=config)    # /v1/models reports the Hub id or run path as given, not the resolved cache path
    demo = a.demo or config.demo
    print(f"serving {run} ({kev.path}) on {kev.device} {a.host}:{a.port}{' with demo routes' if demo else ''}")
    import uvicorn
    uvicorn.run(create_app(kev, demo=demo), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
