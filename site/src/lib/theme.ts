export type Theme = "light" | "dark";

const STORAGE_KEY = "cas-theme";

export function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Storage can be unavailable; the theme still applies for this session.
  }
}

/** Optional: serve NB International Pro from a stylesheet instead of local files. */
export function injectFontStylesheet(): void {
  const href = import.meta.env.VITE_NB_FONT_CSS as string | undefined;
  if (!href) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  document.head.append(link);
}
