"""One error shape for every failure of the server and the Python API: {"error": {"code", "message", "path"?, "retryable"}}.
The codes and statuses are the ones of the Detect contract."""

STATUS = {"bad_request": 422, "too_large": 413, "model_error": 500, "unavailable": 503}


class KevError(Exception):
    def __init__(self, code: str, message: str, path: str | None = None):
        if code not in STATUS: raise ValueError(f"unknown error code {code!r}")
        super().__init__(message)
        self.code, self.message, self.path = code, message, path

    @property
    def status(self) -> int:
        return STATUS[self.code]

    @property
    def retryable(self) -> bool:
        return self.code == "unavailable"

    def body(self) -> dict:
        err = {"code": self.code, "message": self.message, "retryable": self.retryable}
        if self.path: err["path"] = self.path
        return {"error": err}
