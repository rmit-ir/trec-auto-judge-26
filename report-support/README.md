# Report support

Everything behind the paper in sibling repo `../trec-autojudge-26-report`, which holds
only what the paper compiles from (`main.tex`, `sections/`, `tables/`, `figures/*.pdf`,
`references.bib`). Data, scripts, evidence, and notes live here.

| Path | What |
|------|------|
| `render_agreement.py` | Regenerates `figures/agreement-boxplot.pdf` and `figures/judge-agreement.pdf` in the report repo. Reads `docs/figures/*.csv` (hash-checked), which `scripts/agreement_heatmap.py` and `scripts/topic_alpha.py` produce from scored `*.eval.txt` files. Run `python report-support/render_agreement.py` (NumPy, pandas, Matplotlib, Linux Libertine font). |
| `analyze_saved_outputs.py` | Aggregates saved runner `.eval.txt` scores into `evidence/internal-diagnostics.json` (needs SciPy; takes the implementation checkout as argument). Scores only, never answers or caches. |
| `evidence/` | JSON records of hashes and checks (`internal-diagnostics`, `additional-agreement-review`, `figure-reproduction`, `cost-ledger-review`) and the TIRA upload screenshot. |
| `notes/` | `sources.md` (provenance), `gap-analysis.md`, `documentation-review.md`. Author working notes, not paper content. |

Analysis write-up: `docs/judge-agreement-analysis.md`.
