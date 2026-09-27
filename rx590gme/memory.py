"""8 GB of simulated GDDR5 video memory.

The address space is the full 8 GiB of the card, but host memory only backs
64 KiB pages that are allocated *and* have been touched, so the simulator does
not need 8 GiB of host RAM. Touching an unmapped page raises a
:class:`GPUMemoryFault`, just like a VM fault on a real GPU. As on hardware,
protection is page granular: an access that strays past the end of a buffer
but stays inside a mapped page is not caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAGE_SHIFT = 16
PAGE_SIZE = 1 << PAGE_SHIFT
ALLOC_ALIGN = 256
# Nothing is ever mapped below this address, so a null pointer always faults.
HEAP_BASE = 1 << 20


class GPUMemoryFault(RuntimeError):
    """An access touched unmapped or misaligned memory."""


class GPUOutOfMemory(MemoryError):
    pass


@dataclass(frozen=True)
class Buffer:
    """A region of VRAM returned by :meth:`VRAM.alloc`."""

    address: int
    size: int

    def __repr__(self) -> str:
        return f"Buffer(address=0x{self.address:x}, size={self.size})"


class VRAM:
    def __init__(self, size_bytes: int):
        self.size = size_bytes
        self._pages: dict[int, np.ndarray] = {}
        self._page_refs: dict[int, int] = {}
        self._free: list[list[int]] = [[HEAP_BASE, size_bytes - HEAP_BASE]]  # [start, length]
        self._allocs: dict[int, int] = {}

    # ---- allocation ------------------------------------------------------

    @property
    def used_bytes(self) -> int:
        return sum(self._allocs.values())

    @property
    def resident_bytes(self) -> int:
        """Host memory backing pages that have been touched."""
        return len(self._pages) * PAGE_SIZE

    def alloc(self, nbytes: int) -> Buffer:
        if nbytes <= 0:
            raise ValueError("allocation size must be positive")
        size = -(-nbytes // ALLOC_ALIGN) * ALLOC_ALIGN
        for block in self._free:
            start, length = block
            if length >= size:
                block[0] += size
                block[1] -= size
                if block[1] == 0:
                    self._free.remove(block)
                self._allocs[start] = size
                for page in range(start >> PAGE_SHIFT, ((start + size - 1) >> PAGE_SHIFT) + 1):
                    self._page_refs[page] = self._page_refs.get(page, 0) + 1
                return Buffer(start, nbytes)
        raise GPUOutOfMemory(
            f"cannot allocate {nbytes} bytes: {self.used_bytes / 2**30:.2f} GiB of "
            f"{self.size / 2**30:.0f} GiB in use")

    def free(self, buf: Buffer) -> None:
        size = self._allocs.pop(buf.address, None)
        if size is None:
            raise ValueError(f"{buf!r} is not a live allocation")
        for page in range(buf.address >> PAGE_SHIFT, ((buf.address + size - 1) >> PAGE_SHIFT) + 1):
            self._page_refs[page] -= 1
            if self._page_refs[page] == 0:
                del self._page_refs[page]
                self._pages.pop(page, None)
        self._free.append([buf.address, size])
        self._free.sort()
        merged: list[list[int]] = []
        for start, length in self._free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1][1] += length
            else:
                merged.append([start, length])
        self._free = merged

    # ---- device-side access (used by shaders) ----------------------------

    def _page(self, page: int, address: int) -> np.ndarray:
        backing = self._pages.get(page)
        if backing is None:
            if page not in self._page_refs:
                raise GPUMemoryFault(f"page fault at address 0x{address:x} (unmapped)")
            # Mapped but never touched: back it with zeroed host memory now.
            backing = self._pages[page] = np.zeros(PAGE_SIZE // 4, dtype=np.uint32)
        return backing

    @staticmethod
    def _check_aligned(addrs: np.ndarray) -> None:
        if np.any(addrs & np.uint64(3)):
            bad = int(addrs[(addrs & np.uint64(3)) != 0][0])
            raise GPUMemoryFault(f"misaligned dword access at address 0x{bad:x}")

    def read_u32(self, addrs: np.ndarray) -> np.ndarray:
        """Gather one dword from each byte address in ``addrs`` (uint64)."""
        if addrs.size == 0:
            return np.empty(0, dtype=np.uint32)
        self._check_aligned(addrs)
        pages = addrs >> np.uint64(PAGE_SHIFT)
        index = ((addrs & np.uint64(PAGE_SIZE - 1)) >> np.uint64(2)).astype(np.intp)
        first = int(pages[0])
        if np.all(pages == pages[0]):
            return self._page(first, int(addrs[0]))[index]
        out = np.empty(addrs.size, dtype=np.uint32)
        for page in np.unique(pages):
            sel = pages == page
            out[sel] = self._page(int(page), int(addrs[sel][0]))[index[sel]]
        return out

    def write_u32(self, addrs: np.ndarray, values: np.ndarray) -> None:
        """Scatter dwords; when lanes collide the highest lane wins."""
        if addrs.size == 0:
            return
        self._check_aligned(addrs)
        pages = addrs >> np.uint64(PAGE_SHIFT)
        index = ((addrs & np.uint64(PAGE_SIZE - 1)) >> np.uint64(2)).astype(np.intp)
        if np.all(pages == pages[0]):
            self._page(int(pages[0]), int(addrs[0]))[index] = values
            return
        for page in np.unique(pages):
            sel = pages == page
            self._page(int(page), int(addrs[sel][0]))[index[sel]] = values[sel]

    # ---- host-side access (PCIe copies) ----------------------------------

    def write_bytes(self, address: int, data: bytes | bytearray | memoryview) -> None:
        data = memoryview(data).cast("B")
        offset = 0
        while offset < len(data):
            addr = address + offset
            page = self._page(addr >> PAGE_SHIFT, addr).view(np.uint8)
            start = addr & (PAGE_SIZE - 1)
            n = min(PAGE_SIZE - start, len(data) - offset)
            page[start:start + n] = np.frombuffer(data[offset:offset + n], dtype=np.uint8)
            offset += n

    def read_bytes(self, address: int, nbytes: int) -> bytes:
        out = bytearray(nbytes)
        offset = 0
        while offset < nbytes:
            addr = address + offset
            page = self._page(addr >> PAGE_SHIFT, addr).view(np.uint8)
            start = addr & (PAGE_SIZE - 1)
            n = min(PAGE_SIZE - start, nbytes - offset)
            out[offset:offset + n] = page[start:start + n].tobytes()
            offset += n
        return bytes(out)
