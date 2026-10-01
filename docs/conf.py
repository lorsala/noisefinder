# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib.metadata import version

project = 'noisefinder'
copyright = '2026, Lorenzo Sala'
author = 'Lorenzo Sala'
version = version("noisefinder")
release = version

rst_prolog = f"""
.. |version| replace:: {release}
"""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # if you use Google/NumPy-style docstrings
    "sphinx.ext.viewcode",   # optional: adds links to source code
    'sphinx_mdinclude',
    'myst_nb',               # renders .ipynb notebooks (also provides MyST markdown)
]
myst_enable_extensions = [
    "linkify",        # auto-detect bare links
    "colon_fence",    # for ::: fences if used
]

# -- myst-nb (notebook rendering) ---------------------------------------------
# Notebooks are NOT re-executed at build time: the outputs already saved in the
# .ipynb files (as run locally) are rendered as-is. This avoids needing a Jupyter
# kernel / matplotlib in the docs build environment (incl. CI). Keep the notebook
# outputs up to date locally before committing, otherwise the docs will show
# stale results.
nb_execution_mode = "off"
# Don't show stderr output (warnings, tracebacks of caught exceptions, etc.) from
# the notebooks in the built docs.
nb_output_stderr = "remove"

# -- Sync example notebooks for the doc build ----------------------------------
# The tutorial notebooks are authored and run in ../examples/ (with their
# outputs saved from a local run). Sphinx cannot reference files outside its
# source root (docs/), so working copies are kept in docs/examples/. Those
# copies are regenerated automatically from ../examples/ on every build, with
# a couple of build-only fixes applied:
#  - the `%matplotlib ipympl` magic is stripped (incompatible with a static
#    HTML build; irrelevant anyway since notebooks are not re-executed here);
#  - the ipympl interactive-widget output bundle is stripped from saved plot
#    outputs, keeping only the static PNG (the widget can't render statically
#    and, left in, hides the image entirely).
# Just edit/re-run the notebooks in ../examples/ as usual -- the copies here
# are regenerated automatically on the next build, nothing to copy by hand.
import json
import pathlib
import re

_SYNCED_NOTEBOOKS = {
    # filename: markdown H1 title to add if the notebook doesn't start with one
    "pf_howtouse.ipynb": "# How to use (CPSD estimate)",
    "pf_howtouse_decorr.ipynb": None,
}

_WIDGET_MIME_KEYS = (
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.jupyter.widget-state+json",
)

_RST_ROLE_RE = re.compile(r":([a-zA-Z]+):`([^`]+)`")


def _clean_notebook(nb):
    cells = nb.get("cells", [])

    # strip the interactive-backend magic from the first cell, if present
    if cells and cells[0].get("cell_type") == "code":
        src = cells[0]["source"]
        src = "".join(src) if isinstance(src, list) else src
        lines = [
            line for line in src.splitlines()
            if "matplotlib" not in line or "ipympl" not in line
        ]
        cells[0]["source"] = "\n".join(lines)

    # strip non-statically-renderable widget outputs, keep the static PNG
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") != "display_data" or "data" not in out:
                continue
            data = out["data"]
            for key in _WIDGET_MIME_KEYS:
                data.pop(key, None)
            if "image/png" in data and "text/html" in data:
                html = data["text/html"]
                html = "".join(html) if isinstance(html, list) else html
                if "Canvas(" in html or not html.strip():
                    data.pop("text/html", None)

    # convert RST-style cross-reference roles (:class:`...`, :meth:`...`,
    # :math:`...`, ...) to the MyST Markdown equivalent ({class}`...`, etc.),
    # so they render as proper links/formatting instead of literal text.
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell["source"]
        src = "".join(src) if isinstance(src, list) else src
        cell["source"] = [_RST_ROLE_RE.sub(r"{\1}`\2`", src)]
    return nb


_HEADING_RE = re.compile(r"^#+\s+(.+)")


def _ensure_h1_title(cells, title):
    """Make sure the notebook starts with an H1 heading (Sphinx/myst-nb wants
    the first heading to be H1). If the first cell is already a markdown
    heading (any level), promote it to H1 in place rather than adding a
    second, redundant title on top of it."""
    if cells and cells[0].get("cell_type") == "markdown":
        src = cells[0]["source"]
        src = "".join(src) if isinstance(src, list) else src
        m = _HEADING_RE.match(src.strip())
        if m:
            cells[0]["source"] = [f"# {m.group(1).strip()}"]
            return
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": [title]})


def _sync_example_notebooks():
    src_dir = pathlib.Path(__file__).parent.parent / "examples"
    dst_dir = pathlib.Path(__file__).parent / "examples"
    dst_dir.mkdir(exist_ok=True)
    for name, title in _SYNCED_NOTEBOOKS.items():
        src = src_dir / name
        if not src.exists():
            continue
        nb = json.loads(src.read_text(encoding="utf-8"))
        nb = _clean_notebook(nb)
        if title:
            _ensure_h1_title(nb["cells"], title)
        (dst_dir / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")


_sync_example_notebooks()

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Shorten displayed module paths
add_module_names = False


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

import os
import sys

sys.path.insert(0, os.path.abspath('..'))
