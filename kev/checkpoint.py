"""What head.pt says about a checkpoint, as one typed object (read once by kev.evaluate.load, kept on the model as `model.meta`)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadMeta:
    base: str
    base_revision: str | None
    head_dim: int
    option_isolation: bool
    weights_dtype: str | None
    lora: int | None
    temperature: float | None            # the temperature stored in head.pt; None = the checkpoint was never calibrated
    temperature_override: float | None = None   # KEV_TEMPERATURE / Config.temperature

    @classmethod
    def from_dict(cls, meta: dict, temperature_override: float | None = None) -> "HeadMeta":
        t = meta.get("temperature")
        return cls(base=meta["base"], base_revision=meta.get("base_revision"), head_dim=meta.get("head_dim", 256),
                   option_isolation=bool(meta.get("option_isolation", False)), weights_dtype=meta.get("weights_dtype"),
                   lora=meta.get("lora"), temperature=None if t is None else float(t), temperature_override=temperature_override)

    @property
    def applied_temperature(self) -> float:
        """The divisor the pointer head applies to its logits in eval mode (1.0 = raw)."""
        if self.temperature_override is not None: return self.temperature_override
        return 1.0 if self.temperature is None else self.temperature

    @property
    def calibrated(self) -> bool:
        """True when the temperature in use is the one fitted and stored with the checkpoint."""
        return self.temperature is not None and self.temperature_override is None

    def calibration(self) -> dict:
        """One temperature for every question type, fitted on the checkpoint's in-distribution development rows.
        Argmax never changes; `source` says where the applied value comes from."""
        source = "override" if self.temperature_override is not None else "checkpoint" if self.temperature is not None else "none"
        return {"calibrated": self.calibrated, "method": "temperature", "temperature": self.applied_temperature, "source": source}
