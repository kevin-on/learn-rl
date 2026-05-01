import json
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class JSONLMetricsLogger:
    def __init__(self, metrics_path: Path) -> None:
        self.path = metrics_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def write(self, **metrics: Any) -> None:
        record = {key: value for key, value in metrics.items() if value is not None}
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
