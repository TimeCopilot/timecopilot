# Release checklist (v0.0.33)

Depends on **foundationforecast v0.1.6** (TabPFN-3). See the [foundationforecast release checklist](https://github.com/TimeCopilot/foundationforecast/blob/main/docs/RELEASE.md).

## Before tagging

1. Merge PR for TabPFN-3 support into `main`.
2. Confirm `foundationforecast` **0.1.6** is on PyPI.
3. Remove the temporary git source from `pyproject.toml`:

   ```toml
   [tool.uv.sources]
   foundationforecast = { git = "...", branch = "feat/tabpfn-ts-3" }
   ```

4. Refresh the lock file:

   ```bash
   uv lock --upgrade-package foundationforecast
   uv sync
   uv run pytest tests/models/foundation/test_tabpfn.py -m models
   ```

5. Verify version `0.0.33` in `pyproject.toml` and changelog `docs/changelogs/v0.0.33.md`.

## Publish to PyPI

```bash
git checkout main && git pull
git tag v0.0.33
git push origin v0.0.33
```

Ensure `TABPFN_TOKEN` is set in GitHub repository secrets for CI.
