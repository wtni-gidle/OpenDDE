"""Private, job-local resources; never include weights or database caches."""

from contextvars import ContextVar
from functools import wraps
import os
import tempfile

_runtime_directory = ContextVar("opendde_runtime_directory", default=None)


def scratch_base():
    for key in ("SLURM_TMPDIR", "TMPDIR"):
        path = os.environ.get(key)
        if path and os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK):
            return path
    return None


def temporary_directory(*, prefix="opendde-"):
    return tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_base())


def runtime_directory():
    return _runtime_directory.get()


def with_runtime_directory(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with temporary_directory(prefix="opendde-runtime-") as directory:
            token = _runtime_directory.set(directory)
            try:
                return function(*args, **kwargs)
            finally:
                _runtime_directory.reset(token)

    return wrapped
