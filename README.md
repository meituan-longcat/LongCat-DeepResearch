# LongCat-DeepResearch Blog

This directory contains the local static Blog companion to the
LongCat-DeepResearch technical report. It is a preview source, not an
authorization to deploy a public website.

Open `index.html` through a local HTTP server so SVG fragment views work on
mobile layouts. The page has no build step and no external runtime dependency.

Shared visual assets are synchronized from the technical-report project:

- `assets/recorded-overview.png` is the high-resolution, public-benchmark-only
  overview used by the paper's final PDF; its product logos remain sharp at
  desktop width.
- `assets/research-harness-green.svg` includes `#plan`, `#research`, and
  `#compose` views for responsive rendering.
- `assets/data_synthesis_flow.svg` includes `#grounding`, `#validation`, and
  `#trajectory` views.

Links used by the page:

- GitHub: https://github.com/meituan-longcat/LongCat-DeepResearch
- Technical report: https://github.com/meituan-longcat/LongCat-DeepResearch/blob/main/technical_report/LongCat-DeepResearch.pdf
- arXiv: coming soon
- Hugging Face: coming soon

Before a public release, replace the two placeholders with their final URLs
and verify that the GitHub repository and report are publicly accessible.
