"""Bounded MuJoCo executor compiled without Python or the GIL in its loop.

Windows x64 / MuJoCo 3.3.2 ABI only. Native SHA-256 retains the compiled
model and immutable command-payload checks at every applicable boundary.
No JIT compilation, hash-object allocation, logging, or IPC occurs in a step.
"""
import ctypes
import math
import os
from pathlib import Path
import re

import mujoco
import numpy as np
from numba import njit, types
from numba.extending import intrinsic

if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8 or mujoco.mj_version() != 332:
    raise RuntimeError("native wall executor requires Windows x64 and MuJoCo 3.3.2")

_mj = ctypes.CDLL(str(Path(mujoco.__file__).parent / "mujoco.dll"))
_step = _mj.mj_step
_step.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
_step.restype = None
_get_state = _mj.mj_getState
_get_state.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
_get_state.restype = None
_kernel = ctypes.CDLL("kernel32")
_counter = _kernel.QueryPerformanceCounter
_counter.argtypes = (ctypes.c_void_p,)
_counter.restype = ctypes.c_int
_frequency = _kernel.QueryPerformanceFrequency
_frequency.argtypes = (ctypes.c_void_p,)
_frequency.restype = ctypes.c_int
_bcrypt = ctypes.CDLL("bcrypt")
_hash_data = _bcrypt.BCryptHashData
_hash_data.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)
_hash_data.restype = ctypes.c_int32
_finish_hash = _bcrypt.BCryptFinishHash
_finish_hash.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)
_finish_hash.restype = ctypes.c_int32

MODEL_FIELDS = (
    "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "dof_damping", "dof_armature", "dof_frictionloss", "jnt_stiffness",
    "jnt_axis", "jnt_pos", "jnt_range", "geom_pos", "geom_quat", "geom_size",
    "geom_type", "geom_contype", "geom_conaffinity", "actuator_gear",
    "actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_ctrlrange")
ERRORS = {
    1: "NO_VALID_CONTINUATION", 2: "COMMAND_ID_MISMATCH",
    3: "PREDECESSOR_MISMATCH", 4: "INVALID_NATIVE_CLOCK_CONTRACT",
    5: "MISSED_SERVO_WINDOW", 6: "SERVO_STATE_MISMATCH",
    7: "LIVE_MODEL_MISMATCH", 8: "COMMAND_PAYLOAD_MISMATCH",
    9: "NATIVE_REALIZATION_MISMATCH", 10: "NATIVE_HASH_API_ERROR",
    11: "EXECUTION_CANCELLED", 12: "MISSED_PHYSICS_CONSUMPTION_WINDOW",
}


@intrinsic
def _atomic_load(typingctx, array, index):
    signature = types.int64(array, index)
    def codegen(context, builder, sig, args):
        value = context.make_array(sig.args[0])(context, builder, args[0])
        pointer = builder.gep(value.data, [args[1]])
        return builder.load_atomic(pointer, "acquire", 8)
    return signature, codegen


@intrinsic
def _atomic_store(typingctx, array, index, value):
    signature = types.void(array, index, value)
    def codegen(context, builder, sig, args):
        target = context.make_array(sig.args[0])(context, builder, args[0])
        pointer = builder.gep(target.data, [args[1]])
        builder.store_atomic(args[2], pointer, "release", 8)
        return context.get_dummy_value()
    return signature, codegen


@njit(nogil=True)
def atomic_read(array, index=0):
    return _atomic_load(array, index)


@njit(nogil=True)
def atomic_write(array, value, index=0):
    _atomic_store(array, index, value)


@njit(nogil=True)
def now(counter, frequency):
    if not _counter(counter.ctypes.data):
        return math.nan
    return counter[0] / frequency


@njit(nogil=True, fastmath=False)
def release_now(ready, metadata, slot, command, deadline, counter, frequency):
    stamp = now(counter, frequency)
    if not math.isfinite(stamp) or stamp > deadline or _atomic_load(ready, slot) != -1:
        return math.nan
    metadata[slot, 2] = stamp
    _atomic_store(ready, slot, command)
    return stamp


@njit(nogil=True, fastmath=False)
def _model_digest(handle, source, pointers, sizes, doubles, integers, options, digest):
    options[0] = doubles[0]
    options[1] = integers[0]
    options[2] = integers[3]
    options[3] = integers[8]
    options[4] = integers[9]
    options[5] = integers[4]
    options[6] = doubles[3]
    options[7] = doubles[16]
    options[8] = doubles[17]
    if _hash_data(handle, source.ctypes.data, source.size, 0):
        return False
    for i in range(pointers.size):
        if _hash_data(handle, pointers[i], sizes[i], 0):
            return False
    if _hash_data(handle, options.ctypes.data, options.nbytes, 0):
        return False
    if _hash_data(handle, doubles[7:10].ctypes.data, 24, 0):
        return False
    return _finish_hash(handle, digest.ctypes.data, 32, 0) == 0


@njit(nogil=True, fastmath=False)
def _packet_digest(handle, states, torques, tail, digest):
    if _hash_data(handle, states.ctypes.data, states.nbytes, 0):
        return False
    if _hash_data(handle, torques.ctypes.data, torques.nbytes, 0):
        return False
    if _hash_data(handle, tail.ctypes.data, tail.nbytes, 0):
        return False
    return _finish_hash(handle, digest.ctypes.data, 32, 0) == 0


@njit(nogil=True, fastmath=False)
def _same_state(observed, expected):
    for i in range(observed.size):
        if (not math.isfinite(observed[i]) or not math.isfinite(expected[i])
                or abs(observed[i] - expected[i]) > 1e-9):
            return False
    return True


@njit(nogil=True, fastmath=False)
def execute(model_address, data_address, control, state_spec, epoch, count,
            counter, frequency, hash_handle, source, model_pointers, model_sizes,
            option_doubles, option_integers, option_buffer, model_hash, digest,
            states, timings, applied_torques, source_times, slot_states, slot_torques, slot_tail,
            slot_meta, slot_ids, slot_hashes, ready, signals, virtual_times):
    """signals: stop / completed steps / status / active ID."""
    _get_state(model_address, data_address, states[0].ctypes.data, state_spec)
    _atomic_store(signals, 2, 1)
    error = 0
    completed = 0
    for step_index in range(count):
        command = step_index // 10
        substep = step_index % 10
        slot = command % 2
        scheduled = epoch + step_index * .002
        if virtual_times.size:
            started = virtual_times[step_index, 0]
        else:
            started = now(counter, frequency)
            while started < scheduled:
                if _atomic_load(signals, 0):
                    error = 11
                    break
                started = now(counter, frequency)
        if error:
            break
        if _atomic_load(signals, 0):
            error = 11
            break
        timings[step_index, 0] = scheduled
        timings[step_index, 1] = started
        if not math.isfinite(started) or not scheduled <= started < scheduled + .002:
            error = 5
            break
        if substep == 0:
            if _atomic_load(ready, slot) != command:
                error = 1
                break
            if slot_ids[slot, 0] != command:
                error = 2
                break
            if slot_ids[slot, 1] != command - 1:
                error = 3
                break
            if (not math.isfinite(slot_meta[slot, 2])
                    or slot_meta[slot, 2] > slot_meta[slot, 0]
                    or abs(slot_meta[slot, 0] - scheduled) > 1e-9
                    or abs(slot_meta[slot, 1] - scheduled - .020) > 1e-9
                    or abs(states[step_index, 0] - slot_meta[slot, 3]) > 1e-9):
                error = 4
                break
        if not _packet_digest(hash_handle, slot_states[slot], slot_torques[slot], slot_tail[slot], digest):
            error = 10
            break
        if not np.array_equal(digest, slot_hashes[slot]):
            error = 8
            break
        if not _model_digest(hash_handle, source, model_pointers, model_sizes,
                             option_doubles, option_integers, option_buffer, digest):
            error = 10
            break
        if not np.array_equal(digest, model_hash):
            error = 7
            break
        if not _same_state(states[step_index], slot_states[slot, substep]):
            error = 6
            break
        for actuator in range(67):
            control[actuator] = slot_torques[slot, substep, actuator]
        consumed = virtual_times[step_index, 1] if virtual_times.size else now(counter, frequency)
        if not math.isfinite(consumed) or not scheduled <= consumed < scheduled + .002:
            error = 12
            break
        timings[step_index, 2] = consumed
        for actuator in range(67):
            applied_torques[step_index, actuator] = control[actuator]
        if substep == 0:
            _atomic_store(signals, 3, command)
        _step(model_address, data_address)
        _get_state(model_address, data_address, states[step_index + 1].ctypes.data, state_spec)
        realization_matches = _same_state(states[step_index + 1], slot_states[slot, substep + 1])
        finished = consumed if virtual_times.size else now(counter, frequency)
        timings[step_index, 3] = finished
        if substep == 0:
            source_times[command] = finished
        if substep == 9:
            _atomic_store(ready, slot, -1)
        completed = step_index + 1
        _atomic_store(signals, 1, completed)
        if not realization_matches:
            error = 9
            break
    if error:
        _atomic_store(signals, 2, -error)
    else:
        if not virtual_times.size:
            while now(counter, frequency) < epoch + count * .002:
                if _atomic_load(signals, 0):
                    _atomic_store(signals, 2, -11)
                    return completed
        _atomic_store(signals, 2, 2)
    return completed


class NativeHashContext:
    """Own the reusable native hash and checked views of the pinned SDK ABI."""
    def __init__(self, model, source_hash, expected_model_hash):
        header = (Path(mujoco.__file__).parent / "include/mujoco/mjmodel.h").read_text()
        prefix = header.split("struct mjModel_ {", 1)[1].split("mjOption opt;", 1)[0]
        fields = re.findall(r"^\s*(int|size_t)\s+(\w+)\s*;", prefix, re.M)
        class ModelPrefix(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int if kind == "int" else ctypes.c_size_t) for kind, name in fields] + [("option", ctypes.c_double)]
        raw = ModelPrefix.from_address(model._address)
        for _, name in fields:
            if hasattr(model, name) and getattr(raw, name) != getattr(model, name):
                raise RuntimeError(f"MuJoCo model ABI mismatch: {name}")
        option_address = model._address + ModelPrefix.option.offset
        self.doubles = np.ctypeslib.as_array((ctypes.c_double * 31).from_address(option_address))
        self.integers = np.ctypeslib.as_array((ctypes.c_int * 13).from_address(option_address + 248))
        check = (self.doubles[0] == model.opt.timestep and self.doubles[3] == model.opt.tolerance
            and self.doubles[16] == model.opt.density and self.doubles[17] == model.opt.viscosity
            and self.integers[0] == int(model.opt.integrator) and self.integers[3] == int(model.opt.solver)
            and self.integers[4] == model.opt.iterations and self.integers[8] == int(model.opt.disableflags)
            and self.integers[9] == int(model.opt.enableflags)
            and np.array_equal(self.doubles[7:10], model.opt.gravity))
        if not check:
            raise RuntimeError("MuJoCo option ABI mismatch")
        self.views = [getattr(model, field) for field in MODEL_FIELDS]
        self.pointers = np.array([value.ctypes.data for value in self.views], dtype=np.uintp)
        self.sizes = np.array([value.nbytes for value in self.views], dtype=np.uint32)
        self.source = np.frombuffer(source_hash.encode("ascii"), dtype=np.uint8)
        self.expected = np.frombuffer(bytes.fromhex(expected_model_hash), dtype=np.uint8)
        self.option_buffer = np.empty(9)
        self.digest = np.empty(32, dtype=np.uint8)
        self.counter = np.zeros(1, dtype=np.int64)
        self.publication_counter = np.zeros(1, dtype=np.int64)
        frequency = np.zeros(1, dtype=np.int64)
        if not _frequency(frequency.ctypes.data):
            raise RuntimeError("Windows counter frequency unavailable")
        self.frequency = float(frequency[0])
        provider, handle = ctypes.c_void_p(), ctypes.c_void_p()
        open_provider = _bcrypt.BCryptOpenAlgorithmProvider
        open_provider.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        open_provider.restype = ctypes.c_int32
        create_hash = _bcrypt.BCryptCreateHash
        create_hash.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)
        create_hash.restype = ctypes.c_int32
        if open_provider(ctypes.byref(provider), "SHA256", None, 0):
            raise RuntimeError("Windows SHA256 provider unavailable")
        self.provider = provider
        if create_hash(provider, ctypes.byref(handle), None, 0, None, 0, 0x20):
            self.close()
            raise RuntimeError("Windows reusable SHA256 unavailable")
        self.handle = handle.value
        if not _model_digest(self.handle, self.source, self.pointers, self.sizes,
                self.doubles, self.integers, self.option_buffer, self.digest) or not np.array_equal(self.digest, self.expected):
            self.close()
            raise RuntimeError("native SHA256 does not match the original model identity")

    def close(self):
        if getattr(self, "handle", None):
            destroy = _bcrypt.BCryptDestroyHash
            destroy.argtypes = (ctypes.c_void_p,)
            destroy(ctypes.c_void_p(self.handle))
            self.handle = None
        if getattr(self, "provider", None):
            close = _bcrypt.BCryptCloseAlgorithmProvider
            close.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            close(self.provider, 0)
            self.provider = None


class NativeBuffers:
    LAYOUT = {
        "states": ("d", np.float64), "timings": ("d", np.float64),
        "applied_torques": ("d", np.float64),
        "source_times": ("d", np.float64), "slot_states": ("d", np.float64),
        "slot_torques": ("d", np.float64), "slot_tail": ("d", np.float64),
        "slot_meta": ("d", np.float64), "slot_ids": ("q", np.int64),
        "slot_hashes": ("B", np.uint8), "ready": ("q", np.int64), "signals": ("q", np.int64),
    }
    def __init__(self, context, state_size, count):
        self.shapes = {"states": (count + 1, state_size), "timings": (count, 4),
            "applied_torques": (count, 67),
            "source_times": (max(count // 10, 1),), "slot_states": (2, 11, state_size),
            "slot_torques": (2, 10, 67), "slot_tail": (2, 34),
            "slot_meta": (2, 4), "slot_ids": (2, 2), "slot_hashes": (2, 32),
            "ready": (2,), "signals": (4,)}
        self.storage = {name: context.RawArray(self.LAYOUT[name][0], int(np.prod(shape)))
                        for name, shape in self.shapes.items()}
        self.view("ready")[:] = -1
        self.view("signals")[3] = -1

    def view(self, name):
        return np.frombuffer(self.storage[name], dtype=self.LAYOUT[name][1]).reshape(self.shapes[name])

    def stage(self, packet):
        c = packet.certificate
        slot = c.command_id % 2
        if atomic_read(self.view("ready"), slot) != -1:
            raise ValueError("NATIVE_SLOT_STILL_OWNED_BY_EXECUTOR")
        self.view("slot_states")[slot] = packet.integration_states
        self.view("slot_torques")[slot] = packet.torques
        self.view("slot_tail")[slot, :17] = packet.endpoint_velocity
        self.view("slot_tail")[slot, 17:] = packet.reference_end
        self.view("slot_meta")[slot] = (c.execution_start, c.execution_end, math.nan,
                                       c.execution_start_simulation_s)
        self.view("slot_ids")[slot] = (c.command_id, c.predecessor_command_id)
        self.view("slot_hashes")[slot] = np.frombuffer(bytes.fromhex(c.payload_sha256), dtype=np.uint8)
        return slot

    def release(self, packet, publication_time):
        c = packet.certificate
        slot = c.command_id % 2
        if atomic_read(self.view("ready"), slot) != -1:
            raise ValueError("NATIVE_SLOT_STILL_OWNED_BY_EXECUTOR")
        if not math.isfinite(publication_time) or publication_time > c.publish_deadline:
            raise ValueError("LATE_OR_INVALID_PUBLICATION")
        self.view("slot_meta")[slot, 2] = publication_time
        atomic_write(self.view("ready"), c.command_id, slot)

    def publish(self, packet, publication_time):
        self.stage(packet)
        self.release(packet, publication_time)

    def release_on_clock(self, packet, crypto):
        c = packet.certificate
        stamp = release_now(self.view("ready"), self.view("slot_meta"),
            c.command_id % 2, c.command_id, c.publish_deadline,
            crypto.publication_counter, crypto.frequency)
        if not math.isfinite(stamp):
            raise ValueError("LATE_OR_INVALID_PUBLICATION")
        return stamp

    def execute(self, model, data, crypto, epoch, count, virtual_times=None):
        if model.nu != 67:
            raise ValueError("native interface requires the original 67 actuators")
        return execute(model._address, data._address, data.ctrl, int(mujoco.mjtState.mjSTATE_INTEGRATION),
            epoch, count, crypto.counter, crypto.frequency, crypto.handle, crypto.source,
            crypto.pointers, crypto.sizes, crypto.doubles, crypto.integers, crypto.option_buffer,
            crypto.expected, crypto.digest, *(self.view(name) for name in self.LAYOUT),
            np.empty((0, 2)) if virtual_times is None else virtual_times)
