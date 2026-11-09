"""Every KEV_* environment variable that changes how a checkpoint is loaded or served, read once into a frozen object.

    config = Config.from_env()      # the one place that touches os.environ
    kev = Kev.from_pretrained("runs/kev", config=config)

Everything else takes the object. Variables of tooling outside serving stay with their tool: KEV_BASE_URL (tests/test_api.py),
KEV_API (playground/next.config.ts), KEV_GPU, KEV_HF_SECRET and KEV_GIT_COMMIT (modal_app.py, kev/experiment.py), KEV_PROBE_GPU
(docs/modal_probe35.py) and KEV_REMOTE_API_KEY (kev/benchmark.py).
"""
import os
from dataclasses import dataclass, field, fields

DTYPES = ("fp32", "bf16", "fp16")


def _doc(env, text, default, parse=str):
    return field(default=default, metadata={"env": env, "doc": text, "parse": parse})


def _on(raw): return raw == "1"
def _not_off(raw): return raw != "0"


@dataclass(frozen=True)
class Config:
    dtype: str = _doc("KEV_DTYPE", "Backbone dtype: fp32 (exact; every reported number), bf16 or fp16. bf16 halves memory; probabilities move in the third decimal.", "fp32")
    merge: bool = _doc("KEV_MERGE", "Fold the LoRA into the base weights in fp32 before any cast. 0 keeps the adapter separate.", True, _not_off)
    attn: str | None = _doc("KEV_ATTN", "Attention backend. Unset: SDPA on CUDA and on Apple GPUs when serving, eager elsewhere.", None)
    shape_bucket: int = _doc("KEV_SHAPE_BUCKET", "MPS only: pad sequences to a multiple of this many tokens. 1 disables.", 64, int)
    prefix_cache: int = _doc("KEV_PREFIX_CACHE", "Number of state prefixes (KV cache and hidden states) kept between requests. 0 disables.", 4, int)
    prefix_min_tokens: int = _doc("KEV_PREFIX_MIN_TOKENS", "States shorter than this are not cached; below it the branch-only pass is not faster on MPS.", 384, int)
    date_facts: bool = _doc("KEV_DATE_FACTS", "1 appends the day counts between absolute dates in the state (api.with_date_facts). Opt-in preprocessing that changes answers.", False, _on)
    temperature: float | None = _doc("KEV_TEMPERATURE", "Override the checkpoint's fitted temperature. 1.0 gives the raw logits. Unset: use the checkpoint's own.", None, float)
    lora_scale: float = _doc("KEV_LORA_SCALE", "Scale the LoRA at load time (WiSE-FT style interpolation between base 0 and fine-tuned 1). A research knob.", 1.0, float)
    demo: bool = _doc("KEV_DEMO", "1 turns on the demo and experimental routes of kev.serve (same as --demo).", False, _on)
    queue_timeout: float = _doc("KEV_QUEUE_TIMEOUT", "Seconds a request waits for the model before it is refused as unavailable.", 120.0, float)

    def __post_init__(self):
        if self.dtype not in DTYPES: raise ValueError(f"KEV_DTYPE must be one of {DTYPES}, got {self.dtype!r}")
        if self.shape_bucket < 1: raise ValueError("KEV_SHAPE_BUCKET must be at least 1")
        if self.prefix_cache < 0 or self.prefix_min_tokens < 0: raise ValueError("KEV_PREFIX_CACHE and KEV_PREFIX_MIN_TOKENS must not be negative")
        if self.temperature is not None and self.temperature <= 0: raise ValueError("KEV_TEMPERATURE must be positive")

    @classmethod
    def from_env(cls, environ=None) -> "Config":
        env = os.environ if environ is None else environ
        kw = {}
        for f in fields(cls):
            raw = env.get(f.metadata["env"], "")
            if raw == "": continue
            kw[f.name] = f.metadata["parse"](raw)
        return cls(**kw)

    def describe(self) -> dict:
        """The effective settings that can change a number, for GET /v1/models."""
        keep = ("dtype", "merge", "attn", "date_facts", "temperature", "lora_scale", "prefix_cache", "prefix_min_tokens")
        return {k: getattr(self, k) for k in keep}
