"""One bounded immutable result slot, with nonblocking executor reads."""
import pickle
import time
import numpy as np


class SharedResultSlot:
    CAPACITY = 2 * 1024 * 1024

    def __init__(self, context):
        self.storage = context.RawArray("b", self.CAPACITY)
        self.length = context.RawValue("i", 0)
        self.ready = context.RawValue("i", 0)
        self.published_at = context.RawValue("d", 0.)
        self.lock = context.Lock()

    def publish(self, payload):
        raw = pickle.dumps(payload, protocol=5)
        if len(raw) > self.CAPACITY:
            raise ValueError("BOUNDED_PACKET_CAPACITY_EXCEEDED")
        with self.lock:
            if self.ready.value:
                raise ValueError("BOUNDED_PENDING_SLOT_OCCUPIED")
        # Only the planner writes, and only after the previous result was
        # consumed and the executor released the next single request.
        np.frombuffer(self.storage, dtype=np.uint8)[:len(raw)] = np.frombuffer(raw, dtype=np.uint8)
        with self.lock:
            self.length.value = len(raw)
            self.published_at.value = time.perf_counter()
            self.ready.value = 1

    def take(self):
        if not self.lock.acquire(block=False):
            return None
        try:
            if not self.ready.value:
                return None
            raw = np.frombuffer(self.storage, dtype=np.uint8)[:self.length.value].tobytes()
            published = self.published_at.value
            self.ready.value = 0
        finally:
            self.lock.release()
        payload = pickle.loads(raw)
        if "packet" in payload:
            payload["packet"] = payload["packet"].immutable_after_transport()
        return payload, published
