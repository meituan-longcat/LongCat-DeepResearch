# LongCat-DeepResearch Blog

Static companion to the public LongCat-DeepResearch technical report.
The published source is the `gh-pages` branch; its Pages workflow deploys pushes.
Use feature branches and pull requests for updates.

Open `index.html` through a local HTTP server so SVG fragment views work on
mobile layouts. The page has no build step or external runtime dependency.

## Paper figures

The figures come from the public arXiv source for **2609.36071v1** (28 September
2026), downloaded on 4 October 2026 from https://arxiv.org/src/2609.36071.

- `assets/recorded-overview.png` is byte-identical to
  `figs/benchmark-overview/benchmark-scores.png` in that source archive.
- `assets/research-harness-arxiv-2609.36071.svg` is a vector conversion of
  `figs/research-harness-uniform-margins.pdf`.
- `assets/data-synthesis-arxiv-2609.36071.svg` is a vector conversion of
  `figs/data-synthesis-uniform-margins.pdf`.

PDF figures were converted using `pdftocairo -svg`. SVG title/accessibility
metadata and named crop views were added for the existing mobile stage layout;
the desktop figure artwork is unchanged. New filenames avoid stale image caches.
Older figure assets remain available for rollback but are no longer referenced.

## Paper links and citation

- Project: https://meituan-longcat.github.io/LongCat-DeepResearch/
- GitHub: https://github.com/meituan-longcat/LongCat-DeepResearch
- arXiv: https://arxiv.org/abs/2609.36071
- PDF: https://arxiv.org/pdf/2609.36071
- Hugging Face: https://huggingface.co/papers/2609.36071

The Citation section uses the official BibTeX from
https://arxiv.org/bibtex/2609.36071, including the complete author list.
