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

Both 2400px diagrams were rendered from the arXiv figure PDFs and visually
inspected. GitHub Markdown API rendering passed for both READMEs, including
image references and the benchmark width. Citations match official arXiv BibTeX
after whitespace normalization; local references exist. git diff --check and
open_source_preflight.py passed. The initial raw-HTML text assertion did not
account for syntax-highlighting spans; the parser-based check passed. Remote
content checks follow the merge.

### Commit Hash

`90d31d1` — Sync bilingual READMEs and figures with arXiv report.

## 2026-10-23 — Improve README discoverability

### Question

Make the public READMEs clearer about what LongCat-DeepResearch is and how
planning, independent section research, and report revision work.

### Analysis / Root Cause

The tagline and overview were accurate but general, so readers and searches had
little descriptive text identifying the project as a multi-agent deep research
harness. The harness stages were documented, but the rationale for separate
section contexts and global-to-local revision was not stated in one place.

### Solution

Rewrite the tagline and first overview sentence around an open-source multi-agent
deep research harness. Keep the existing structure, add a short "Research
Planning and Report Revision" subsection after the harness stages, link the
canonical arXiv report, and mirror the change in Chinese. No figures, benchmark
numbers, or performance claims changed.

### Files Changed

- `README.md`
- `README_zh-CN.md`
- `docs/dev.md`

### Validation

Both READMEs have matching taglines, overview sentences, and subsection
structure. The English subsection is under 100 words. git diff --check passed.

### Commit Hash

Containing commit: `Clarify research planning and report revision in bilingual README`.

Planner verification (2026-10-23): checked release scope against README and source; desktop 1440px/mobile 390px rendering and image loading pass. Repository metadata now includes description, six relevant topics and the project homepage.
