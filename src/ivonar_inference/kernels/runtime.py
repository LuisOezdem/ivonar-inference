from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from ctypes import POINTER, byref, c_char, c_char_p, c_float, c_int, c_size_t, c_uint, c_void_p
from pathlib import Path

import torch

CACHE_DIR = Path.home() / ".ivonar" / "kernels"
_COMPILE_OPTIONS = ("-default-device", "--std=c++17")
_LIMIT_PERSISTING_L2 = 6
_ATTRIBUTE_MAX_PERSISTING_L2 = 108
_ATTRIBUTE_MAX_ACCESS_WINDOW = 109
_STREAM_ATTRIBUTE_ACCESS_WINDOW = 1
_ACCESS_PROPERTY_NORMAL = 0
_ACCESS_PROPERTY_PERSISTING = 2


class KernelError(RuntimeError):
    pass


class AccessWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", c_void_p),
        ("num_bytes", c_size_t),
        ("hit_ratio", c_float),
        ("hit_property", c_int),
        ("miss_property", c_int),
    ]


class _AttributeValue(ctypes.Union):
    _fields_ = [("padding", c_char * 64), ("window", AccessWindow)]


def _library(candidates: Sequence[str]) -> ctypes.CDLL:
    errors: list[str] = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise KernelError("no CUDA library could be loaded: " + "; ".join(errors))


class _Nvrtc:
    def __init__(self) -> None:
        major = str(torch.version.cuda or "12").split(".")[0]
        torch_lib = Path(torch.__file__).parent / "lib"
        if sys.platform == "win32":
            name = f"nvrtc64_{major}0_0.dll"
            if hasattr(os, "add_dll_directory") and torch_lib.is_dir():
                os.add_dll_directory(str(torch_lib))
            candidates = [name, str(torch_lib / name)]
        else:
            candidates = [f"libnvrtc.so.{major}", "libnvrtc.so"]
            for nvidia_lib in Path(torch.__file__).parent.parent.glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*"):
                candidates.append(str(nvidia_lib))
        lib = _library(candidates)
        lib.nvrtcVersion.argtypes = [POINTER(c_int), POINTER(c_int)]
        lib.nvrtcCreateProgram.argtypes = [POINTER(c_void_p), c_char_p, c_char_p, c_int, POINTER(c_char_p), POINTER(c_char_p)]
        lib.nvrtcCompileProgram.argtypes = [c_void_p, c_int, POINTER(c_char_p)]
        lib.nvrtcGetProgramLogSize.argtypes = [c_void_p, POINTER(c_size_t)]
        lib.nvrtcGetProgramLog.argtypes = [c_void_p, c_char_p]
        lib.nvrtcGetCUBINSize.argtypes = [c_void_p, POINTER(c_size_t)]
        lib.nvrtcGetCUBIN.argtypes = [c_void_p, c_char_p]
        lib.nvrtcDestroyProgram.argtypes = [POINTER(c_void_p)]
        lib.nvrtcGetErrorString.argtypes = [c_int]
        lib.nvrtcGetErrorString.restype = c_char_p
        self.lib = lib
        major_out, minor_out = c_int(), c_int()
        self._check(lib.nvrtcVersion(byref(major_out), byref(minor_out)))
        self.version = (major_out.value, minor_out.value)

    def _check(self, result: int) -> None:
        if result != 0:
            message = self.lib.nvrtcGetErrorString(result)
            raise KernelError(f"NVRTC error: {message.decode() if message else result}")

    def compile(self, source: str, name: str, options: Sequence[str]) -> bytes:
        program = c_void_p()
        self._check(self.lib.nvrtcCreateProgram(byref(program), source.encode(), name.encode(), 0, None, None))
        try:
            encoded = [option.encode() for option in options]
            array = (c_char_p * len(encoded))(*encoded)
            status = self.lib.nvrtcCompileProgram(program, len(encoded), array)
            size = c_size_t()
            self._check(self.lib.nvrtcGetProgramLogSize(program, byref(size)))
            log = ctypes.create_string_buffer(size.value)
            self._check(self.lib.nvrtcGetProgramLog(program, log))
            if status != 0:
                raise KernelError(f"kernel compilation failed:\n{log.value.decode(errors='replace')}")
            self._check(self.lib.nvrtcGetCUBINSize(program, byref(size)))
            binary = ctypes.create_string_buffer(size.value)
            self._check(self.lib.nvrtcGetCUBIN(program, binary))
            return binary.raw
        finally:
            self.lib.nvrtcDestroyProgram(byref(program))


class _Driver:
    def __init__(self) -> None:
        candidates = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "libcuda.so"]
        lib = _library(candidates)
        lib.cuInit.argtypes = [c_uint]
        lib.cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
        lib.cuDeviceGet.argtypes = [POINTER(c_int), c_int]
        lib.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
        lib.cuCtxSetCurrent.argtypes = [c_void_p]
        lib.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_void_p]
        lib.cuModuleUnload.argtypes = [c_void_p]
        lib.cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
        lib.cuLaunchKernel.argtypes = [
            c_void_p, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_void_p, c_void_p, c_void_p,
        ]
        lib.cuDeviceGetAttribute.argtypes = [POINTER(c_int), c_int, c_int]
        lib.cuCtxSetLimit.argtypes = [c_int, c_size_t]
        lib.cuStreamSetAttribute.argtypes = [c_void_p, c_int, POINTER(_AttributeValue)]
        self.lib = lib
        self.check(lib.cuInit(0))
        self._contexts: dict[int, c_void_p] = {}
        self._bound: dict[int, int] = {}
        self._persisting: dict[int, int] = {}
        self._lock = threading.Lock()

    def check(self, result: int) -> None:
        if result != 0:
            message = c_char_p()
            self.lib.cuGetErrorString(result, byref(message))
            raise KernelError(f"CUDA driver error {result}: {message.value.decode() if message.value else 'unknown'}")

    def bind(self, device_index: int) -> None:
        thread = threading.get_ident()
        if self._bound.get(thread) == device_index:
            return
        with self._lock:
            context = self._contexts.get(device_index)
            if context is None:
                device = c_int()
                self.check(self.lib.cuDeviceGet(byref(device), device_index))
                context = c_void_p()
                self.check(self.lib.cuDevicePrimaryCtxRetain(byref(context), device))
                self._contexts[device_index] = context
        self.check(self.lib.cuCtxSetCurrent(context))
        self._bound[thread] = device_index

    def load_module(self, binary: bytes) -> c_void_p:
        module = c_void_p()
        buffer = ctypes.create_string_buffer(binary, len(binary))
        self.check(self.lib.cuModuleLoadData(byref(module), buffer))
        return module

    def get_function(self, module: c_void_p, name: str) -> c_void_p:
        function = c_void_p()
        self.check(self.lib.cuModuleGetFunction(byref(function), module, name.encode()))
        return function

    def device_attribute(self, device_index: int, attribute: int) -> int:
        value = c_int()
        self.check(self.lib.cuDeviceGetAttribute(byref(value), attribute, device_index))
        return int(value.value)

    def persisting_capacity(self, device_index: int) -> int:
        try:
            limit = self.device_attribute(device_index, _ATTRIBUTE_MAX_PERSISTING_L2)
            window = self.device_attribute(device_index, _ATTRIBUTE_MAX_ACCESS_WINDOW)
        except KernelError:
            return 0
        return max(0, min(limit, window))

    def reserve_persisting(self, device_index: int, nbytes: int) -> None:
        with self._lock:
            if self._persisting.get(device_index, -1) >= nbytes:
                return
        self.bind(device_index)
        self.check(self.lib.cuCtxSetLimit(_LIMIT_PERSISTING_L2, nbytes))
        with self._lock:
            self._persisting[device_index] = nbytes

    def persist_on_stream(self, stream: int, base: int, nbytes: int) -> None:
        value = _AttributeValue()
        value.window = AccessWindow(base, nbytes, 1.0, _ACCESS_PROPERTY_PERSISTING, _ACCESS_PROPERTY_NORMAL)
        self.check(self.lib.cuStreamSetAttribute(c_void_p(stream), _STREAM_ATTRIBUTE_ACCESS_WINDOW, byref(value)))


_nvrtc: _Nvrtc | None = None
_driver: _Driver | None = None
_modules: dict[str, "Module"] = {}
_state_lock = threading.Lock()


def nvrtc() -> _Nvrtc:
    global _nvrtc
    with _state_lock:
        if _nvrtc is None:
            _nvrtc = _Nvrtc()
        return _nvrtc


def driver() -> _Driver:
    global _driver
    with _state_lock:
        if _driver is None:
            _driver = _Driver()
        return _driver


class Kernel:
    __slots__ = ("driver", "function", "name")

    def __init__(self, owner: _Driver, function: c_void_p, name: str) -> None:
        self.driver = owner
        self.function = function
        self.name = name


class Launch:
    __slots__ = ("kernel", "grid", "block", "shared", "_args", "_params")

    def __init__(
        self,
        kernel: Kernel,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Iterable[object],
        shared: int = 0,
    ) -> None:
        self.kernel = kernel
        self.grid = tuple(int(value) for value in grid)
        self.block = tuple(int(value) for value in block)
        self.shared = int(shared)
        self._args = tuple(args)
        self._params = (c_void_p * len(self._args))(*[ctypes.addressof(arg) for arg in self._args])

    def __call__(self, stream: int) -> None:
        self.kernel.driver.check(
            self.kernel.driver.lib.cuLaunchKernel(
                self.kernel.function,
                self.grid[0], self.grid[1], self.grid[2],
                self.block[0], self.block[1], self.block[2],
                self.shared, c_void_p(stream), self._params, None,
            )
        )


class Module:
    def __init__(self, owner: _Driver, handle: c_void_p, key: str) -> None:
        self.driver = owner
        self.handle = handle
        self.key = key
        self._kernels: dict[str, Kernel] = {}

    def kernel(self, name: str) -> Kernel:
        kernel = self._kernels.get(name)
        if kernel is None:
            kernel = Kernel(self.driver, self.driver.get_function(self.handle, name), name)
            self._kernels[name] = kernel
        return kernel


def _architecture(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _cache_key(source: str, defines: Mapping[str, object], architecture: str, version: tuple[int, int]) -> str:
    digest = hashlib.sha256()
    digest.update(source.encode())
    for name in sorted(defines):
        digest.update(f"\n-D{name}={defines[name]}".encode())
    digest.update(f"\n{architecture}\nnvrtc{version[0]}.{version[1]}".encode())
    return digest.hexdigest()[:24]


def _read_cache(key: str) -> bytes | None:
    path = CACHE_DIR / f"{key}.cubin"
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _write_cache(key: str, binary: bytes) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_DIR / f"{key}.{os.getpid()}.tmp"
        temporary.write_bytes(binary)
        temporary.replace(CACHE_DIR / f"{key}.cubin")
    except OSError:
        pass


def compile_module(source: str, defines: Mapping[str, object], device: torch.device, name: str = "ivonar") -> Module:
    if device.type != "cuda":
        raise KernelError("ternary kernels need a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.init()
    owner = driver()
    owner.bind(index)
    compiler = nvrtc()
    architecture = _architecture(torch.device("cuda", index))
    key = _cache_key(source, defines, architecture, compiler.version)
    with _state_lock:
        module = _modules.get(key)
    if module is not None:
        return module
    binary = _read_cache(key)
    if binary is None:
        options = [f"--gpu-architecture={architecture}", *_COMPILE_OPTIONS]
        options.extend(f"-D{item}={value}" for item, value in sorted(defines.items()))
        binary = compiler.compile(source, f"{name}.cu", options)
        _write_cache(key, binary)
    module = Module(owner, owner.load_module(binary), key)
    with _state_lock:
        _modules.setdefault(key, module)
        return _modules[key]


def pointer(tensor: torch.Tensor | None) -> c_void_p:
    return c_void_p(0 if tensor is None else tensor.data_ptr())


def int32(value: int) -> c_int:
    return c_int(int(value))


def float32(value: float) -> ctypes.c_float:
    return ctypes.c_float(float(value))
