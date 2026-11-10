"""Which CPUs to pin pool workers to (Linux sysfs; standard library only, safe to import anywhere)."""
import os
from pathlib import Path
from typing import Dict, List, Tuple

_SYS = Path("/sys/devices/system/cpu")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _expand(cpulist: str) -> List[int]:
    out = []
    for part in cpulist.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def physical_cores() -> List[Tuple[int, int]]:
    """One (cpu_id, l3_domain) per physical core among the CPUs this process may run on: the lowest
    CPU id of each SMT sibling group, with the id of the lowest CPU sharing its L3 as the domain."""
    allowed = sorted(os.sched_getaffinity(0))
    seen, cores = set(), []
    for cpu in allowed:
        sib = tuple(c for c in _expand(_read(_SYS / f"cpu{cpu}/topology/thread_siblings_list")) if c in allowed) or (cpu,)
        if sib in seen:
            continue
        seen.add(sib)
        l3 = _expand(_read(_SYS / f"cpu{cpu}/cache/index3/shared_cpu_list")) or [cpu]
        cores.append((min(sib), min(l3)))
    return cores


def spread_cpus(n: int) -> List[int]:
    """n distinct physical-core CPU ids, taken round-robin across L3 domains so any prefix of the
    list is balanced over the L3 caches (and, on NPS1 EPYC, over the memory channels). Raises if the
    process has fewer than n physical cores."""
    by_l3: Dict[int, List[int]] = {}
    for cpu, l3 in physical_cores():
        by_l3.setdefault(l3, []).append(cpu)
    if n > sum(map(len, by_l3.values())):
        raise ValueError(f"asked for {n} workers but only {sum(map(len, by_l3.values()))} physical cores are available")
    order, groups = [], [by_l3[k] for k in sorted(by_l3)]
    for i in range(max(map(len, groups))):
        order.extend(g[i] for g in groups if i < len(g))
    return order[:n]
