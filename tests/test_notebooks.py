"""Static validation of the Colab notebooks.

These do not execute the notebooks — they catch the mistakes that waste a Colab
session: invalid JSON, syntax errors hidden behind the notebook format, leaked
tokens, missing model ids, or a "token notebook" that never uses a token.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(ROOT.glob("*.ipynb"))
DEFAULT_NB = ROOT / "Herogod_Colab.ipynb"
TOKEN_NB = ROOT / "Herogod_Colab_HF_Token.ipynb"
BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "ScottzillaSystems/qwen-image-edit-plus-nsfw-lora"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def source_of(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def code_cells(notebook: dict) -> list[str]:
    return [source_of(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def python_cells(notebook: dict) -> list[str]:
    return [cell for cell in code_cells(notebook) if not cell.lstrip().startswith("%%")]


def strip_magics(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            lines.append("pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def test_notebooks_exist():
    assert DEFAULT_NB.is_file(), "the default Colab notebook is missing"
    assert TOKEN_NB.is_file(), "the Hugging Face token notebook is missing"
    assert len(NOTEBOOKS) == 2, f"unexpected notebooks: {[path.name for path in NOTEBOOKS]}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_is_valid_nbformat(path):
    notebook = load(path)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"].get("accelerator") == "GPU"
    assert len(notebook["cells"]) >= 8
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, index
        assert source_of(cell).strip(), f"empty cell {index}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_python_cells_compile(path):
    notebook = load(path)
    for index, source in enumerate(python_cells(notebook)):
        try:
            compile(strip_magics(source), f"{path.name}:cell-{index}", "exec")
        except SyntaxError as error:  # pragma: no cover - failure path
            pytest.fail(f"{path.name} cell {index} has a syntax error: {error}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_use_the_same_model(path):
    notebook = load(path)
    text = "\n".join(source_of(cell) for cell in notebook["cells"])
    assert BASE_MODEL in text
    assert LORA_REPO in text


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_no_hardcoded_tokens_or_outputs(path):
    raw = path.read_text()
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", raw), "a Hugging Face token literal is committed"
    notebook = load(path)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == [], "notebook cells must not carry saved outputs"
            assert cell.get("execution_count") is None


def test_token_notebook_reads_and_never_prints_the_token():
    notebook = load(TOKEN_NB)
    code = "\n".join(code_cells(notebook))

    # it must source the token from somewhere explicit
    assert "userdata.get(\"HF_TOKEN\")" in code
    assert "getpass.getpass" in code
    assert 'os.environ["HF_TOKEN"] = HF_TOKEN' in code

    # and it must never echo it
    assert "print(HF_TOKEN)" not in code
    assert "print(f\"{HF_TOKEN}" not in code
    assert "print(f'{HF_TOKEN}" not in code
    fingerprints = [match for match in re.findall(r"print\(f?[^\n]*\{HF_TOKEN[^\n]*\)", code)]
    assert fingerprints == [], f"these lines could print the token: {fingerprints}"

    # it validates access before downloading, with the token
    assert "--check-access" in code
    assert "scripts/download_models.py" in code
    # the studio token is distinct from the HF token
    assert 'HEROGOD_API_TOKEN' in code


def test_token_notebook_orders_verification_before_download():
    notebook = load(TOKEN_NB)
    cells = code_cells(notebook)
    check_index = next(index for index, cell in enumerate(cells) if "--check-access" in cell)
    download_index = next(
        index for index, cell in enumerate(cells) if '"--dest", "/content/models"' in cell
    )
    assert check_index < download_index, "access must be verified before the 40 GB download"


def test_token_notebook_clears_the_token_at_the_end():
    notebook = load(TOKEN_NB)
    code = "\n".join(code_cells(notebook))
    assert 'os.environ.pop("HF_TOKEN", None)' in code


def test_download_script_supports_token_flows():
    script = (ROOT / "scripts" / "download_models.py").read_text()
    assert "--check-access" in script
    assert "HF_TOKEN" in script
    assert "GatedRepoError" in script or "gated" in script.lower()
    assert "hf_transfer" not in script  # transfer is enabled via env, not the script
