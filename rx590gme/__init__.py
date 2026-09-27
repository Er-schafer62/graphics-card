"""A software replica of the AMD Radeon RX 590 GME graphics card.

    >>> from rx590gme import RX590GME
    >>> gpu = RX590GME()
    >>> gpu
    <AMD Radeon RX 590 GME: 36 CUs, 8 GB GDDR5, 1380 MHz>
"""

from .device import LaunchStats, RX590GME, occupancy, pack_args
from .isa import AssemblerError, Program, assemble
from .kernels import builtin_kernel, builtin_kernel_names
from .memory import Buffer, GPUMemoryFault, GPUOutOfMemory
from .specs import RX590_GME, GPUSpec, spec_sheet
from .wavefront import GPUExecutionError, GPUHangError

__all__ = [
    "RX590GME", "RX590_GME", "GPUSpec", "LaunchStats", "Buffer", "Program",
    "AssemblerError", "GPUMemoryFault", "GPUOutOfMemory", "GPUHangError", "GPUExecutionError",
    "assemble", "builtin_kernel", "builtin_kernel_names", "occupancy", "pack_args", "spec_sheet",
]
__version__ = "0.1.0"
