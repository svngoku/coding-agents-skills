# Frontend Performance Deep Dive

## Core Web Vitals (what users actually feel)

| Metric | Measures | Budget |
|--------|----------|--------|
| LCP | Largest contentful paint — loading | < 2.5 s |
| INP | Interaction to next paint — responsiveness | < 200 ms |
| CLS | Cumulative layout shift — stability | < 0.1 |

Measure with `web-vitals` in-app (real-user monitoring), Lighthouse for lab/regression
checks, and DevTools for debugging the waterfall.

## Bundle strategy

1. **Measure** — `webpack-bundle-analyzer`, `rollup-plugin-visualizer`, or Vite `--debug`.
2. **Cut** — remove dead deps, per-module imports (`lodash/merge` not `lodash`),
   tree-shakeable ESM builds, `"sideEffects": false`.
3. **Split** — route-level `React.lazy`/`import()`, vendor chunk splitting, `preload` the
   most likely next route's chunk.

Enforce with `size-limit` or `bundlesize` — fail CI when gzipped initial JS exceeds the
budget (common target: 170 KB).

## Image pipeline

- **Format**: AVIF (best) → WebP → JPEG fallback; PNG only for transparency.
- **Delivery**: CDN resizing (`?w=640&format=avif`) or build-time transform (sharp,
  `next/image`, `astro:assets`).
- **Responsive**: `srcset` + `sizes` so each viewport fetches the right file:

```html
<img
  srcset="hero-320w.webp 320w, hero-640w.webp 640w, hero-1280w.webp 1280w"
  sizes="(max-width: 640px) 320px, 1280px"
  width="1280" height="720"        <!-- prevents CLS -->
  loading="lazy" fetchpriority="high"
  alt="Product hero"
/>
```

`loading="lazy"` below the fold, `fetchpriority="high"` for the LCP image, always set
dimensions or `aspect-ratio` to prevent layout shift.

## CSS/JS delivery & caching headers

- **Critical CSS**: inline above-the-fold styles in `<head>`; load the rest async
  (preload + onload swap or the `media="print"` trick).
- **JS**: `defer` all scripts; `async` only when execution order doesn't matter.
- **Fonts**: `font-display: swap`, preload the woff2, subset glyphs.
- **Caching**: hashed assets `Cache-Control: public, max-age=31536000, immutable`; HTML
  `no-cache` so deploys propagate.

## React rendering performance

### When re-renders are the problem

Profile with React DevTools Profiler first. If a 1,000-row list re-renders when typing in
an unrelated input, fix the data flow, not the memoization:

```tsx
// BAD: new array identity every render -> every row re-renders
const rows = items.map((i) => ({ ...i, label: fmt(i) }));
// GOOD: compute only when inputs change
const rows = useMemo(() => items.map((i) => ({ ...i, label: fmt(i) })), [items]);
```

### memo / useMemo / useCallback

- `React.memo(Component)` — skip re-render when props are referentially equal.
- `useCallback` — stable function identity for props passed to memoized children.
- `useMemo` — cache expensive derived values; don't wrap cheap computations — the memo
  itself costs a comparison.
- React Compiler (experimental) auto-memoizes and removes most manual memo work.

### List virtualization

```tsx
import { FixedSizeList } from "react-window";

<FixedSizeList height={600} width="100%" itemCount={10000} itemSize={40}>
  {({ index, style }) => <div style={style}>Row {index}</div>}
</FixedSizeList>
```

Use for 100+ rows, infinite scroll, or logs. Keep rows memoized and stable-sized.

### Avoiding layout thrash

Batch reads then writes — interleaving forces a synchronous reflow per access:

```js
// BAD: read/write/read/write -> 4 reflows
els.forEach((el) => { const w = el.offsetWidth; el.style.width = w + 1 + "px"; });
// GOOD: read all, then write all -> 1 reflow
const widths = els.map((el) => el.offsetWidth);
els.forEach((el, i) => { el.style.width = widths[i] + 1 + "px"; });
```

Prefer `transform`/`opacity` animations (compositor thread) over `top`/`width` (layout).

## Measurement workflow for a frontend fix

1. Lighthouse / DevTools trace → pick the biggest opportunity (usually images, then
   render-blocking CSS/JS).
2. Apply one change; re-run the identical audit.
3. Confirm with real-user data (`web-vitals`) after deploy; budgets on CI catch regressions.
