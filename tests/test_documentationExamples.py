"""Run the Python examples of the README files, so the documentation stays executable.

Every ```python block of a README is executed in order in one namespace (later blocks
may use names defined by earlier ones), inside a temporary working directory.
"""
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ["README.md", "tool/README.md"]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def pythonBlocks(relativePath):
    return BLOCK.findall((ROOT / relativePath).read_text(encoding="utf-8"))


@pytest.mark.parametrize("document", DOCUMENTS)
def test_pythonExamplesRun(document, tmp_path, monkeypatch):
    blocks = pythonBlocks(document)
    assert blocks, f"no python examples found in {document}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOOLBASE_SECURED_ENABLED", raising=False)
    namespace = {"__name__": f"docs_{Path(document).parent.name or 'root'}"}
    for index, code in enumerate(blocks):
        try:
            exec(compile(code, f"{document}[block {index}]", "exec"), namespace)
        except Exception as exc:  # show which block failed
            pytest.fail(f"{document} block {index} failed: {type(exc).__name__}: {exc}\n{code}")


def test_referencedFilesExist():
    """Relative links and image sources in the READMEs point at existing files."""
    link = re.compile(r"\]\(([^)#]+?)\)|srcset=\"([^\"]+)\"|src=\"([^\"]+)\"")
    for document in DOCUMENTS + ["regressionHandler/README.md", "fieldMapping/README.md"]:
        base = (ROOT / document).parent
        text = (ROOT / document).read_text(encoding="utf-8")
        for match in link.finditer(text):
            target = next(g for g in match.groups() if g)
            if "://" in target:
                continue
            assert (base / target).exists(), f"{document}: missing {target}"
