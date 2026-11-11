import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def json_file_lock(path: str | Path, *, shared: bool = False) -> Iterator[None]:
    """Coordinate cross-process access to a JSON cache file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f"{target.name}.lock")
    with lock_path.open("a+") as lock_file:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(lock_file.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_locked_json(path: str | Path) -> Any:
    target = Path(path)
    with json_file_lock(target, shared=True):
        with target.open("r", encoding="utf-8") as input_file:
            return json.load(input_file)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON without exposing a partially written destination file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with json_file_lock(target):
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_file:
                temp_path = Path(output_file.name)
                json.dump(payload, output_file)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temp_path, target)
            temp_path = None
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
