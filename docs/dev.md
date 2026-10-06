# Development Log

## 2026-10-06 — Refresh repository README and clean merged branches

### Question

Remove obsolete branches and update GitHub README figures to match the public
arXiv report, keeping English and Chinese documentation consistent.

### Analysis / Root Cause

Two blog feature branches were fully merged into gh-pages, with no unique
commits. The main branch still referenced earlier harness and data-construction
figures and an abbreviated citation. Its benchmark image already matches arXiv.
The benchmark image also used inline styling that GitHub does not retain.

### Solution

Delete the two fully merged remote feature branches after verifying their exact
heads and ancestry. Retain main and gh-pages. Render both updated diagram PDFs
from the arXiv 2609.36071v1 source at 2400px into the existing approved PNG paths.
Update both READMEs with the arXiv paper badge, full official BibTeX, public score
table, precise comparison wording, and GitHub-supported image width markup.
Keep the already matching benchmark image and the bundled PDF unchanged.

Deleted branch heads (recoverable from gh-pages history):

- feat/blog-arxiv-figures-citation-20261004: f7c62091ce4c55dbe3003ff5aac155586bb0c2a4
- feat/blog-single-report-link-20261005: 705a076ba0881bafde4a8d4f4457741fd7a09fd9

Figure provenance:

- Source: https://arxiv.org/src/2609.36071 (v1, 28 September 2026)
- Harness: figs/research-harness-uniform-margins.pdf
- Data construction: figs/data-synthesis-uniform-margins.pdf
- Rendering: pdftoppm -singlefile -scale-to 2400 -png
- Benchmark PNG SHA-256: f7b66863a8819045b3d78adece1257989626635005231445c16ef32e5f93ca0a

### Files Changed

- README.md
- README_zh-CN.md
- assets/harness_overview.png
- assets/data_construction.png
- docs/dev.md

### Verification

Pending image, Markdown reference, citation, and release preflight checks.

### Commit Hash

Pending.
