# pip install timecopilot: Diagnosis Report

**Date:** 2026-02-12
**Environment:** Python 3.13.0, pip 24.2, macOS 15 (Darwin 25.2.0), Apple Silicon (arm64)
**Tested version:** timecopilot 0.0.23

---

## Summary

`pip install timecopilot` fails on Python 3.13 after ~6 minutes of dependency resolution. The root cause is pip's backtracking resolver exhausting all 37 available versions of `optuna` (pulled in transitively via `mlforecast`) until it reaches `optuna 2.3.0`, which is only available as a source tarball. Building that tarball fails because its `setup.py` imports `pkg_resources`, which was removed from Python 3.13's standard library. For users on slower networks, the 5+ minute resolution phase with no visible progress appears as an indefinite "stall."

---

## Test Setup

An isolated venv was created completely separate from the project's dev environment:

```
~/timecopilot-pip-diagnosis/.venv/   (test — Python 3.13 + pip 24.2)
~/Documents/GitHub/timecopilot/.venv/ (dev — Python 3.11 + uv, untouched)
```

The install was run with triple-verbose logging and timestamps:

```bash
pip install timecopilot -vvv 2>&1 | \
  while IFS= read -r line; do printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | \
  tee ~/timecopilot-pip-diagnosis/install-log.txt
```

---

## Timeline

| Phase | Timestamp | Duration | Description |
|-------|-----------|----------|-------------|
| Start | 08:29:58 | — | pip 24.2 begins resolving `timecopilot` |
| Metadata fetch | 08:29:58 – 08:30:34 | 36s | Downloads `.whl.metadata` for timecopilot 0.0.23 and all direct dependencies |
| Dependency resolution | 08:30:34 – 08:35:47 | 5m 13s | Scans every wheel on PyPI for 200+ transitive packages, evaluating compatibility tags |
| Backtracking begins | 08:35:47 | — | pip emits: "This is taking longer than usual... See pip.pypa.io/warnings/backtracking" |
| optuna version sweep | 08:36:02 – 08:36:12 | 10s | Tries optuna 4.7.0, 4.6.0, 4.5.0, ... down to 2.3.0 (37 versions total) |
| optuna 2.3.0 sdist build | 08:36:12 – 08:36:18 | 6s | Downloads 258 KB tarball, installs setuptools 82.0.0, runs `get_requires_for_build_wheel` |
| **Failure** | 08:36:18 | — | `ModuleNotFoundError: No module named 'pkg_resources'` |

**Total wall time:** 6 minutes 20 seconds
**Log size:** 173,670 lines

Output was flowing continuously throughout (~28,000 lines/minute), so there was no true network stall. However, without `-vvv`, pip shows **zero output** during the 5-minute resolution phase, which is indistinguishable from a hang.

---

## Root Cause Analysis

### The dependency chain

```
timecopilot 0.0.23
  -> mlforecast>=1.0.2
       -> optuna          (NO version constraint)
```

`mlforecast` declares `optuna` as a dependency with no lower or upper bound. This gives pip's resolver maximum freedom to try any version.

### The backtracking trigger

pip first resolved `optuna 4.7.0` successfully, but a conflict elsewhere in the dependency tree (likely involving `scipy<=1.15.3` or another constrained package) forced pip to backtrack. During backtracking, pip tried every optuna version in descending order:

```
4.7.0 -> 4.6.0 -> 4.5.0 -> 4.4.0 -> 4.3.0 -> 4.2.1 -> 4.2.0 -> 4.1.0 -> 4.0.0
-> 3.6.2 -> 3.6.1 -> 3.6.0 -> 3.5.1 -> 3.5.0 -> 3.4.1 -> 3.4.0 -> 3.3.0 -> 3.2.0
-> 3.1.1 -> 3.1.0 -> 3.0.6 -> ... -> 3.0.0 -> 2.10.1 -> ... -> 2.4.0 -> 2.3.0
```

Versions 2.4.0+ have `.whl` files (prebuilt wheels). Version 2.3.0 is **source-only** (`.tar.gz`).

### The build failure

When pip attempted to build `optuna 2.3.0` from source:

1. Downloaded `optuna-2.3.0.tar.gz` (258 KB)
2. Installed build dependency `setuptools 82.0.0`
3. Ran `get_requires_for_build_wheel` which executed optuna's `setup.py`
4. `setup.py` line 7: `import pkg_resources` -> **`ModuleNotFoundError`**

`pkg_resources` was part of `setuptools` historically but Python 3.12+ removed `distutils` from the stdlib, and `setuptools 82.0.0` no longer implicitly provides `pkg_resources` in the build isolation environment the way old `setup.py`-based packages expect.

### Why this appears as a "stall" to users

Without `-vvv`, pip's default output during dependency resolution is:

```
Collecting timecopilot
  Downloading timecopilot-0.0.23-py3-none-any.whl.metadata (16 kB)
```

Then **nothing** for 5+ minutes while it scans 173,000+ package links. On slower networks or PyPI mirrors, this phase takes even longer. Users interpret the silence as a hang and kill the process.

---

## Why uv Works

The project's dev environment uses `uv` with a lockfile (`uv.lock`, 8,546 lines) that pins `optuna==4.5.0`. Key differences:

1. **Lockfile-based resolution:** `uv` resolves from `uv.lock` — no backtracking needed
2. **Faster resolver:** Even without a lockfile, uv's resolver is significantly faster than pip's and avoids the pathological backtracking behavior
3. **Better Python 3.13 compatibility:** uv handles the `pkg_resources` removal gracefully by not falling back to ancient source distributions

---

## Dependency Scale

The full transitive dependency tree includes **200+ packages**. Notable heavy dependencies:

| Category | Packages |
|----------|----------|
| ML frameworks | torch, jax, jaxlib, lightning, pytorch-lightning, ray |
| Forecasting | neuralforecast, statsforecast, mlforecast, prophet, gluonts |
| Data | pandas, numpy, scipy, scikit-learn, pyarrow, datasets |
| Foundation models | transformers, tokenizers, safetensors, huggingface-hub |
| Observability | opentelemetry-*, logfire, wandb, tensorboard |
| AI/LLM | openai, anthropic, pydantic-ai, mistralai, cohere, groq |
| HTTP | httpx, aiohttp, requests, grpcio |

This massive dependency tree amplifies pip's resolution time because it must evaluate compatibility for every package against every platform tag.

---

## Recommendations

### Short-term fixes

1. **Add a lower bound on optuna** in `pyproject.toml`:
   ```toml
   # In mlforecast's dependencies or as an explicit constraint
   optuna>=3.0
   ```
   This prevents pip from backtracking into source-only versions. Since timecopilot already pins `mlforecast>=1.0.2` and the codebase uses modern optuna APIs, `optuna>=3.0` is safe.

2. **Add `setuptools` to build dependencies** for any sdist-based installs, or ensure all transitive deps provide wheels for Python 3.13.

### Medium-term fixes

3. **Publish a `constraints.txt`** alongside the package that pins known-good versions of heavy transitive dependencies, so `pip install timecopilot -c constraints.txt` avoids backtracking.

4. **Document `uv` as the recommended installer** in the README and getting-started guide:
   ```bash
   # Recommended
   uv pip install timecopilot

   # Alternative (slower, may timeout on Python 3.13)
   pip install timecopilot
   ```

### Long-term fixes

5. **Reduce the dependency footprint** — consider making heavy ML frameworks (torch, jax, ray, wandb) optional extras:
   ```toml
   [project.optional-dependencies]
   torch = ["torch>=2.4.0", "pytorch-lightning>=2.0.0"]
   all = ["timecopilot[torch,jax,ray]"]
   ```
   This would dramatically reduce install time for users who only need a subset of forecasting backends.

6. **File an issue upstream** on `mlforecast` to add `optuna>=3.0` as a lower bound, since optuna < 3.0 is incompatible with Python 3.12+.

---

## Raw Data

- **Log file:** 173,670 lines, captured with per-line timestamps
- **Total packages scanned:** 200+ unique package names on PyPI
- **optuna versions tried:** 37 (4.7.0 down to 2.3.0)
- **Backtracking warnings emitted:** 3
- **Final error:** `ModuleNotFoundError: No module named 'pkg_resources'` in `optuna-2.3.0/setup.py`
- **Nothing was installed** — the venv contained only pip 24.2 after the failed attempt
