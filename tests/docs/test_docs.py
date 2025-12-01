import re
import sys
from pathlib import Path

import pytest
from mktestdocs import check_md_file


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    [p for p in Path("docs").rglob("*.md") if "changelogs" not in p.parts],
    ids=str,
)
@pytest.mark.flaky(reruns=3, reruns_delay=80)
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.docs
@pytest.mark.flaky(reruns=3, reruns_delay=80)
def test_readme():
    check_md_file("README.md", memory=True)


@pytest.mark.docs
def test_latest_changelog():
    def version_key(filename):
        match = re.search(r"(\d+\.\d+\.\d+)", str(filename))
        if match:
            version_string = match.group(1)
            return tuple(map(int, version_string.split(".")))
        return (0, 0, 0)

    changelog_dir = Path("docs/changelogs")
    changelogs = sorted(changelog_dir.glob("v*.md"), key=version_key)
    latest_changelog = changelogs[-1] if changelogs else None
    check_md_file(latest_changelog, memory=True)


@pytest.mark.docs
@pytest.mark.flaky(reruns=3, reruns_delay=80)
@pytest.mark.parametrize(
    "fpath",
    Path("timecopilot").glob("**/*.py"),
    ids=str,
)
def test_py_examples(fpath):
    check_md_file(fpath=fpath, memory=True)


skip_gift_eval_mark = pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="gift-eval notebook not supported on Python 3.13",
)


def maybe_skip_gift_eval(fpath):
    out = str(fpath)
    if out == "docs/examples/gift-eval.ipynb":
        out = pytest.param(out, marks=skip_gift_eval_mark)
    return out


# skipping notebooks for now, as they rise a no space error
# see: https://github.com/TimeCopilot/timecopilot/actions/runs/18858375517/job/53811527062?pr=245
# @pytest.mark.docs
# @pytest.mark.parametrize(
#    "fpath",
#    [maybe_skip_gift_eval(f) for f in Path("docs").rglob("*.ipynb")],
# )
# def test_notebooks(fpath):
#    nb = nbformat.read(fpath, as_version=4)
#    client = NotebookClient(nb, timeout=600, kernel_name="python3")
#    client.execute()
