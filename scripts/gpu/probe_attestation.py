"""Ask the GPU's driver for its identity and attestation data, and print what comes back.

Runs on the pod. It calls NVIDIA's management library (NVML) for the GPU's
UUID and serial number, the confidential computing (CC) state, the device
certificate chain, and an attestation report over a fresh 32-byte nonce. On a
GPU outside CC mode, the last calls are expected to fail; this records how.

Example:
    /opt/upstream/.venv/bin/python scripts/gpu/probe_attestation.py
"""

from __future__ import annotations

import os

import pynvml as nv


def attempt(name, fn):
    try:
        result = fn()
    except Exception as e:  # report every failure, since failures are the point
        print(f"{name}: failed: {type(e).__name__}: {e}")
        return None
    print(f"{name}: {result}")
    return result


def describe(obj) -> str:
    fields = getattr(obj, "_fields_", None)
    if not fields:
        return repr(obj)
    parts = []
    for field in fields:
        value = getattr(obj, field[0])
        if isinstance(value, (bytes, bytearray)) or hasattr(value, "__len__") and not isinstance(value, str):
            parts.append(f"{field[0]}=<{len(value)} bytes>")
        else:
            parts.append(f"{field[0]}={value}")
    return ", ".join(parts)


def main() -> None:
    nv.nvmlInit()
    print(f"driver: {nv.nvmlSystemGetDriverVersion()}")
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    attempt("name", lambda: nv.nvmlDeviceGetName(handle))
    attempt("UUID", lambda: nv.nvmlDeviceGetUUID(handle))
    attempt("serial", lambda: nv.nvmlDeviceGetSerial(handle))
    attempt("CC state", lambda: describe(nv.nvmlSystemGetConfComputeState()))
    attempt("CC settings", lambda: describe(nv.nvmlSystemGetConfComputeSettings()))
    attempt("CC GPU ready state", lambda: nv.nvmlSystemGetConfComputeGpusReadyState())
    attempt("CC certificate", lambda: describe(nv.nvmlDeviceGetConfComputeGpuCertificate(handle)))
    nonce = os.urandom(32)
    attempt("CC attestation report", lambda: describe(nv.nvmlDeviceGetConfComputeGpuAttestationReport(handle, nonce)))
    nv.nvmlShutdown()


if __name__ == "__main__":
    main()
