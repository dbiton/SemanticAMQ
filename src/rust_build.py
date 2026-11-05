"""Builds the Rust kernels (CapCoverRustFilter/, crate capcover_rs) into the running interpreter.

paper.py calls ensure_built() before it imports anything that needs capcover_rs, so a run always
uses kernels compiled from the checked-out sources. It runs `maturin develop --release`, which is
incremental: a few seconds when nothing changed, minutes the first time. Standard library only,
because it runs before numpy/faiss are imported."""
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CRATE_DIR = Path(__file__).resolve().parent.parent / "CapCoverRustFilter"


def _importable() -> bool:
    return importlib.util.find_spec("capcover_rs") is not None


def _maturin_command() -> list:
    """`python -m maturin`, installing maturin with pip first if the interpreter lacks it."""
    if importlib.util.find_spec("maturin") is None:
        logger.info("installing maturin")
        subprocess.run([sys.executable, "-m", "pip", "install", "maturin>=1.5,<2"], check=True)
    return [sys.executable, "-m", "maturin"]


def ensure_built() -> None:
    """Compile and install capcover_rs for this interpreter. If it cannot be built (no Rust
    toolchain, no sources) but an earlier build is installed, warns and carries on with that one;
    otherwise stops with the command to run."""
    if not (CRATE_DIR / "Cargo.toml").exists() or shutil.which("cargo") is None:
        reason = "no CapCoverRustFilter/ sources" if not CRATE_DIR.exists() else "cargo not found (install Rust: https://rustup.rs)"
        if _importable():
            logger.warning(f"not rebuilding the Rust kernels ({reason}); using the installed capcover_rs")
            return
        raise SystemExit(f"capcover_rs is not installed and cannot be built: {reason}")
    env = dict(os.environ)
    if sys.prefix != sys.base_prefix:               # maturin develop installs into the active virtualenv
        env["VIRTUAL_ENV"] = sys.prefix
    logger.info("building the Rust kernels (maturin develop --release)")
    done = subprocess.run(_maturin_command() + ["develop", "--release"], cwd=CRATE_DIR, env=env)
    if done.returncode != 0:
        if _importable():
            logger.warning("Rust build failed; using the previously installed capcover_rs")
            return
        raise SystemExit("building capcover_rs failed (see the maturin output above)")
