# Development Log

## 2026-10-05 — Keep one technical report entry

### Question

Keep only one of the PDF and Technical report buttons.

### Analysis / Root Cause

Both hero buttons led to the same paper through different arXiv endpoints.

### Solution

Remove the separate PDF button and retain Technical report linked to the arXiv
abstract, which also provides PDF access.

### Files Changed

- `index.html`
- `docs/dev.md`

### Verification

The hero contains exactly one paper entry, and the Technical report destination
is unchanged. git diff --check passed. Live verification follows deployment.

### Commit Hash

`0d9a3c0` — Keep one technical report button in blog header.

## 2026-10-04 — Synchronize arXiv figures and citation

### Question

The arXiv paper has redrawn figures, while the blog still uses old artwork.
Update the matching blog figures and Citation. The provided Overleaf project
was inspected read-only but returned a Restricted page in the current browser.

### Analysis / Root Cause

The blog referenced old native SVG artwork. The public arXiv source provides
updated harness and data-synthesis PDFs. The benchmark PNG is already identical.
The citation omitted most authors and lacked arXiv fields.

### Solution

Convert the arXiv source PDFs to vector SVGs, add matching mobile stage views,
and update the figure references with new asset filenames. Preserve the matching
benchmark image. Replace the citation with official arXiv BibTeX and add a direct
PDF link. Document provenance and update article modification metadata.

### Files Changed

- `index.html`
- `README.md`
- `assets/research-harness-arxiv-2609.36071.svg`
- `assets/data-synthesis-arxiv-2609.36071.svg`
- `docs/dev.md`

### Verification

git diff --check passed. Local HTML assets, six SVG stage views and aspect
ratios, and JSON metadata were validated. Citation matches official arXiv BibTeX
after whitespace normalization. Original SVG drawing/definition elements are
unchanged from pdftocairo output. Both full figures and all six mobile crops
were rendered and visually inspected. The benchmark PNG matches the source
SHA-256. Live asset/content verification follows Pages deployment.

### Commit Hash

`6ceef71` — Sync blog figures and full citation with arXiv report.
