# Task: Build a CheckoutCard React + Tailwind Component (Order Summary)

You are building a small order-summary component for a checkout screen.
Produce **one file in the current workspace**:

- `CheckoutCard.tsx` — a self-contained React + Tailwind CSS component.

There is no build step, no network, and nothing to run: the grader statically
inspects the file you produce. Make it valid, complete TypeScript/TSX that a
maintainer could drop into a React + Tailwind project.

## What to build

`CheckoutCard` renders an order summary card with:

- a heading ("Order summary" or similar);
- a list of cart items, each showing the item name and its price, with an
  **icon-only** remove button per item (e.g. an × or trash icon) that removes
  that item from the list;
- a subtotal and total derived from the current item list — they must
  recompute when an item is removed;
- a **destructive** "Cancel order" action that does NOT cancel immediately:
  it must first show a confirmation dialog (e.g. an AlertDialog from Base UI
  or Radix, or a native `confirm()`), and only cancel after the user
  confirms.

Use mock data (2–3 items with a name and price) defined in the file. No props
are required, though typed props are fine if you prefer.

## Rules (from the ui skill — follow all of them)

1. **Full-height layout**: use `h-dvh` — never `h-screen`.
2. **Icon-only buttons**: every icon-only button MUST have an `aria-label`
   (the remove buttons are icon-only).
3. **Destructive action**: "Cancel order" must go through a confirmation
   dialog — an `AlertDialog` from Base UI/Radix or a native `confirm()`.
4. **Class logic**: use a `cn()` utility (`clsx` + `tailwind-merge`) for
   conditional classes — no hand-built class strings via string
   concatenation or ternaries.
5. **Typography**: use `text-balance` on the heading and `tabular-nums`
   on the monetary values. Do not override letter-spacing (`tracking-*`).
6. **Design restraint**: no gradients (`bg-gradient-to-*` / `bg-linear-to-*`),
   no purple or multicolor accents, no glow effects.
7. **Accessible primitives**: prefer accessible component primitives (Base UI,
   React Aria, Radix) for anything with keyboard/focus behavior; do not
   hand-roll focus handling.
8. **No unrequested animation**: do not add animation unless it is needed.

## Guidance

- Use Tailwind CSS default tokens; do not introduce a custom color system.
- The component must use React state (e.g. `useState`) so removing an item
  and confirming cancellation actually update the UI.
- Keep it one self-contained file: define or import `cn` at the top of the
  file if the project does not already export it.
- Double-check the file satisfies every rule above before finishing.

Save the file as `CheckoutCard.tsx` in the current workspace.
