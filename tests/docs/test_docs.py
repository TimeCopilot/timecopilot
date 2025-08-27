import re
from pathlib import Path

import nbformat
import pytest
from mktestdocs import check_md_file
from nbclient import NotebookClient


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    [p for p in Path("docs").rglob("*.md") if "changelogs" not in p.parts],
    ids=str,
)
@pytest.mark.flaky(reruns=3, reruns_delay=5)
def test_docs(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.docs
@pytest.mark.flaky(reruns=3, reruns_delay=5)
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
@pytest.mark.flaky(reruns=3, reruns_delay=5)
@pytest.mark.parametrize(
    "fpath",
    Path("timecopilot").glob("**/*.py"),
    ids=str,
)
def test_py_examples(fpath):
    check_md_file(fpath=fpath, memory=True)


@pytest.mark.docs
@pytest.mark.parametrize(
    "fpath",
    Path("docs").rglob("*.ipynb"),
    ids=str,
)
def test_notebooks(fpath):
    nb = nbformat.read(fpath, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
