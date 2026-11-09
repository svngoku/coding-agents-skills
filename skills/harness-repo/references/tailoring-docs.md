# Tailoring the docs/ knowledge base

The scaffold writes complete, honest-for-day-0 docs. Tailor them so they describe *this*
project. Prefer a visible `TODO` over invented detail.

| Doc | Tailor by |
|-----|-----------|
| `ARCHITECTURE.md` | Rename layers to the real ones; redraw the Mermaid layer and trust-boundary diagrams with real components; record stack choices as ADRs |
| `DESIGN.md` | Real naming/format rules from the linter config; API and component patterns that actually apply; delete sections that do not |
| `PRODUCT_SENSE.md` | User's purpose statement; 3+ concrete non-goals (ask the user) |
| `QUALITY_SCORE.md` | Keep honest grades; add real domains once code exists |
| `RELIABILITY.md` | Mark which critical paths exist; keep SLO table as TODO until tiers are known |
| `SECURITY.md` | Follow `references/security.md` "Tailoring" steps |
| `PLANS.md` | Already links the initial plan; add rows as plans are created |

Rules: keep all diagrams Mermaid; keep links relative and resolving (GP-6); do not let any doc
exceed what an agent will actually read; run the golden-principles linter (`node .opencode/tools/golden_principles.js`, or `python3 ...golden_principles.py` on the python stack) after edits.

When the user has a first feature in mind, create `docs/exec-plans/active/<date>-<slug>.md`
from `docs/exec-plans/_template.md` and add it to `PLANS.md`.
