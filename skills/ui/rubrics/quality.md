# Quality rubric — CheckoutCard React + Tailwind order summary

Score the agent's `CheckoutCard.tsx` **0.0–1.0**. Anchor: 1.0 = a component
a frontend maintainer would merge unchanged. The deterministic grader verifies
that the required constructs exist; this rubric scores the quality of the
implementation. Award partial credit proportionally per criterion.

## Criteria

1. **Accessibility (30%)** — every icon-only button has a descriptive
   `aria-label`; the confirmation dialog is the product of an accessible
   primitive (focus is trapped and Escape closes via the primitive, not
   hand-rolled code); visible `focus-visible` styles are present on
   interactive elements; decorative SVGs are `aria-hidden`; color contrast
   on prices and actions is sufficient.

2. **Interaction correctness (30%)** — "Cancel order" never cancels
   immediately: it opens a real confirmation dialog (AlertDialog or
   `confirm()`), and the cart is cleared only after explicit confirmation.
   Remove-item buttons update the list, and subtotal/total recompute from
   state. No dead buttons or state that never changes.

3. **Restraint (20%)** — no gradients, no purple/multicolor accents, no glow
   effects; no unrequested animation; no `tracking-*` letter-spacing
   tweaks; Tailwind default tokens only; full-height layout uses `h-dvh`
   (never `h-screen`; a card with no full-height utility at all is worth
   about half credit here).

4. **Skill-rule consistency (20%)** — conditional classes go through a
   `cn()` utility (`clsx` + `tailwind-merge`); `text-balance` on the
   heading and `tabular-nums` on monetary values; no `useEffect` for
   render logic; accessible primitives handle keyboard/focus behavior; the
   file is clean, readable, and self-contained.

## Penalties

- **−20%** if the file would not compile or render as-is (invalid TSX,
  missing imports, missing state).
- **−10%** if the component is a copy-paste of a generic card with the rule
  surface area but no working interactions.
