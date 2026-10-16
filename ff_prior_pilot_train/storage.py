"""Bounded retries for Windows transient file sharing; never retry physics."""
import time
from ff_prior import storage as base


def retry_write(operation, *args, **kwargs):
    for attempt in range(8):
        try:
            return operation(*args, **kwargs)
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.02 * 2**attempt, .5))


def atomic_json(path, value):
    return retry_write(base.atomic_json, path, value)


def atomic_npz(path, **arrays):
    return retry_write(base.atomic_npz, path, **arrays)


class Ledger(base.Ledger):
    def save(self):
        atomic_json(self.path, self.data)
