---
name: ui
description: >
  Opinionated constraints for building better web interfaces with coding agents.
  Use this skill whenever the user asks you to build, review, or fix a UI —
  landing pages, dashboards, components, or any frontend code — and whenever
  you write Tailwind CSS, motion/react animations, accessible component
  primitives (Base UI, React Aria, Radix), or review agent-generated interface
  code for quality, accessibility, and performance. Also trigger for "make the
  UI better", "review my interface", "clean up this component", or any frontend
  work in React, Next.js, or plain HTML/CSS.
---

# UI Skills

When invoked, apply these opinionated constraints for building better interfaces.

## How to use

- `/ui-skills`  
  Apply these constraints to any UI work in this conversation.

- `/ui-skills <file>`  
  Review the file against all constraints below and output:
  - violations (quote the exact line/snippet)
  - why it matters (1 short sentence)
  - a concrete fix (code-level suggestion)

## Stack

- MUST use Tailwind CSS defaults unless custom values already exist or are explicitly requested
- MUST use `motion/react` (formerly `framer-motion`) when JavaScript animation is required
- SHOULD use `tw-animate-css` for entrance and micro-animations in Tailwind CSS
- MUST use `cn` utility (`clsx` + `tailwind-merge`) for class logic

## Components

- MUST use accessible component primitives for anything with keyboard or focus behavior (`Base UI`, `React Aria`, `Radix`)
- MUST use the project’s existing component primitives first
- NEVER mix primitive systems within the same interaction surface
- SHOULD prefer [`Base UI`](https://base-ui.com/react/components) for new primitives if compatible with the stack
- MUST add an `aria-label` to icon-only buttons
- NEVER rebuild keyboard or focus behavior by hand unless explicitly requested

## Interaction

- MUST use an `AlertDialog` for destructive or irreversible actions
- SHOULD use structural skeletons for loading states
- NEVER use `h-screen`, use `h-dvh`
- MUST respect `safe-area-inset` for fixed elements
- MUST show errors next to where the action happens
- NEVER block paste in `input` or `textarea` elements

## Animation

- NEVER add animation unless it is explicitly requested
- MUST animate only compositor props (`transform`, `opacity`)
- NEVER animate layout properties (`width`, `height`, `top`, `left`, `margin`, `padding`)
- SHOULD avoid animating paint properties (`background`, `color`) except for small, local UI (text, icons)
- SHOULD use `ease-out` on entrance
- NEVER exceed `200ms` for interaction feedback
- MUST pause looping animations when off-screen
- SHOULD respect `prefers-reduced-motion`
- NEVER introduce custom easing curves unless explicitly requested
- SHOULD avoid animating large images or full-screen surfaces

## Typography

- MUST use `text-balance` for headings and `text-pretty` for body/paragraphs
- MUST use `tabular-nums` for data
- SHOULD use `truncate` or `line-clamp` for dense UI
- NEVER modify `letter-spacing` (`tracking-*`) unless explicitly requested

## Layout

- MUST use a fixed `z-index` scale (no arbitrary `z-*`)
- SHOULD use `size-*` for square elements instead of `w-*` + `h-*`

## Performance

- NEVER animate large `blur()` or `backdrop-filter` surfaces
- NEVER apply `will-change` outside an active animation
- NEVER use `useEffect` for anything that can be expressed as render logic

## Design

- NEVER use gradients unless explicitly requested
- NEVER use purple or multicolor gradients
- NEVER use glow effects as primary affordances
- SHOULD use Tailwind CSS default shadow scale unless explicitly requested
- MUST give empty states one clear next action
- SHOULD limit accent color usage to one per view
- SHOULD use existing theme or Tailwind CSS color tokens before introducing new ones

## Anti-Patterns to Avoid

The most common agent-generated UI mistakes — flag these immediately:

- **Gradient backgrounds everywhere** — never use gradients unless explicitly requested, and never purple or multicolor gradients
- **`h-screen` full-viewport layouts** — breaks on mobile URL bars; use `h-dvh`
- **Animating layout properties** — `width`, `height`, `top`, `left` cause reflows; animate only `transform` and `opacity`
- **Hand-rolled focus & keyboard behavior** — rebuilding what accessible primitives already provide
- **Icon-only buttons without `aria-label`** — invisible to screen readers
- **Blocking paste in inputs** — breaks password managers and legitimate users
- **Custom easing curves and `tracking-*` tweaks** — unrequested style noise that fights the design system
- **`will-change` and large `blur()` everywhere** — GPU memory and paint cost with no benefit
- **Ignoring `prefers-reduced-motion`** — motion must be optional, never mandatory

## When to Use / Not Use

**Use this skill when:**

- Building new UI from scratch (components, pages, dashboards, forms)
- Reviewing agent-generated or existing frontend code for quality, accessibility, or performance
- Adding animations, loading states, or interactive elements
- Working with Tailwind CSS, motion/react, or accessible component primitives

**Do NOT use this skill when:**

- The project has an established design system with conflicting conventions — the project's system wins
- The stack has no Tailwind and no component library (e.g., plain server-rendered HTML) — still apply the accessibility and interaction principles, but skip class-level rules
- The task is backend, data, or infrastructure work with no UI surface
- The ask is visual direction or creative design (brand work, image generation) — this skill defines constraints, not aesthetics
