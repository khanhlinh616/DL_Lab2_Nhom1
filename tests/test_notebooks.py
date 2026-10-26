"""
Structural tests for Jupyter notebooks to ensure validity, execution, and absence of hardcoded absolute paths.
Does NOT execute notebooks to ensure fast test runs.
"""

import os
import re
import pytest
import nbformat


def get_notebook_paths():
    """Returns list of all .ipynb files in the notebooks directory."""
    nb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")
    if not os.path.exists(nb_dir):
        return []
    return [
        os.path.join(nb_dir, f)
        for f in os.listdir(nb_dir)
        if f.endswith(".ipynb")
    ]


@pytest.mark.parametrize("nb_path", get_notebook_paths())
def test_notebook_valid_structure(nb_path):
    """Verify notebook adheres to valid nbformat schema."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    nbformat.validate(nb)


@pytest.mark.parametrize("nb_path", get_notebook_paths())
def test_notebook_no_error_outputs(nb_path):
    """Verify that no code cell in the notebook has error outputs."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                output_type = output.get("output_type", "")
                if output_type == "error":
                    ename = output.get("ename", "Error")
                    evalue = output.get("evalue", "")
                    pytest.fail(f"Cell {idx} in {os.path.basename(nb_path)} produced error: {ename}: {evalue}")


@pytest.mark.parametrize("nb_path", get_notebook_paths())
def test_notebook_executed_cells(nb_path):
    """Verify that all code cells have been executed (execution_count is not None)."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) > 0, f"Notebook {nb_path} contains no code cells."
    for idx, cell in enumerate(code_cells):
        assert cell.get("execution_count") is not None, f"Code cell {idx} in {os.path.basename(nb_path)} has not been executed."


@pytest.mark.parametrize("nb_path", get_notebook_paths())
def test_notebook_no_absolute_paths_in_source(nb_path):
    """Verify notebook sources do not contain hardcoded local absolute paths (e.g. C:\\, /Users/, /home/)."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    forbidden_patterns = [
        re.compile(r"[a-zA-Z]:\\"),
        re.compile(r"/Users/"),
        re.compile(r"/home/"),
    ]

    for idx, cell in enumerate(nb.cells):
        source = cell.get("source", "")
        for pattern in forbidden_patterns:
            match = pattern.search(source)
            if match:
                pytest.fail(
                    f"Cell {idx} in {os.path.basename(nb_path)} contains forbidden absolute path pattern: '{match.group()}'"
                )


@pytest.mark.parametrize("nb_path", get_notebook_paths())
def test_notebook_file_size_under_limit(nb_path):
    """Verify notebook file size is under 5 MB limit."""
    size_bytes = os.path.getsize(nb_path)
    size_mb = size_bytes / (1024 * 1024)
    assert size_mb < 5.0, f"Notebook {os.path.basename(nb_path)} size is {size_mb:.2f} MB (exceeds 5 MB limit)."
