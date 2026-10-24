import type { MermaidConfig } from "mermaid";

const MAX_SOURCE_LENGTH = 50_000;
let nextId = 0;
let renderQueue = Promise.resolve();

function diagramConfig(): MermaidConfig {
  const style = getComputedStyle(document.documentElement);
  const color = (name: string) => style.getPropertyValue(`--${name}`).trim();
  const foreground = color("foreground");
  const card = color("card");
  const muted = color("muted");
  const accent = color("accent");
  const primary = color("primary");

  return {
    startOnLoad: false,
    securityLevel: "strict",
    suppressErrorRendering: true,
    maxTextSize: MAX_SOURCE_LENGTH,
    maxEdges: 500,
    theme: "base",
    look: "classic",
    layout: "dagre",
    fontFamily: style.fontFamily,
    htmlLabels: false,
    flowchart: { htmlLabels: false },
    sequence: { useMaxWidth: false },
    er: { useMaxWidth: false },
    themeVariables: {
      darkMode: document.documentElement.classList.contains("dark"),
      fontFamily: style.fontFamily,
      fontSize: "14px",
      background: card,
      primaryColor: accent,
      primaryTextColor: foreground,
      primaryBorderColor: primary,
      secondaryColor: muted,
      secondaryTextColor: foreground,
      secondaryBorderColor: primary,
      tertiaryColor: card,
      tertiaryTextColor: foreground,
      tertiaryBorderColor: primary,
      lineColor: foreground,
      textColor: foreground,
      mainBkg: accent,
      edgeLabelBackground: card,
      actorBkg: accent,
      actorBorder: primary,
      actorTextColor: foreground,
      actorLineColor: color("muted-foreground"),
      signalColor: foreground,
      signalTextColor: foreground,
      labelBoxBkgColor: muted,
      labelBoxBorderColor: primary,
      labelTextColor: foreground,
      loopTextColor: foreground,
      noteBkgColor: muted,
      noteBorderColor: primary,
      noteTextColor: foreground,
      activationBkgColor: muted,
      activationBorderColor: primary,
      sequenceNumberColor: card,
      attributeBackgroundColorOdd: muted,
      attributeBackgroundColorEven: card,
    },
  };
}

/** Enhance fences without loading Mermaid for documents containing only code. */
export function renderMermaidWithin(root: HTMLElement): () => void {
  const blocks = [
    ...root.querySelectorAll<HTMLElement>("pre > code.language-mermaid"),
  ].filter((code) => !code.closest(".mermaid-diagram"));
  if (!blocks.length) return () => {};

  const diagrams = blocks.map((code, index) => {
    const pre = code.parentElement!;
    let heading = pre.previousElementSibling;
    while (heading && !/^H[1-6]$/.test(heading.tagName)) {
      heading = heading.previousElementSibling;
    }
    const label = heading?.textContent?.trim()
      ? `${heading.textContent.trim()} diagram`
      : `Mermaid diagram ${index + 1}`;
    const source = code.textContent ?? "";
    const figure = document.createElement("figure");
    figure.className = "mermaid-diagram";
    const status = document.createElement("p");
    status.className = "mermaid-status";
    status.setAttribute("role", "status");
    const viewport = document.createElement("div");
    viewport.className = "mermaid-viewport";
    viewport.tabIndex = 0;
    viewport.setAttribute("role", "region");
    viewport.setAttribute("aria-label", label);
    viewport.hidden = true;
    const canvas = document.createElement("div");
    canvas.className = "mermaid-canvas";
    viewport.append(canvas);
    const details = document.createElement("details");
    details.className = "mermaid-source";
    const summary = document.createElement("summary");
    summary.textContent = "View Mermaid source";
    pre.setAttribute("role", "region");
    pre.setAttribute("aria-label", `${label} source`);
    details.append(summary);
    figure.append(status, viewport, details);
    pre.replaceWith(figure);
    details.append(pre);
    return { pre, figure, status, viewport, canvas, details, source, label };
  });

  let disposed = false;
  let revision = 0;
  let dark = document.documentElement.classList.contains("dark");

  function render() {
    const current = ++revision;
    for (const diagram of diagrams) {
      const stale = () =>
        disposed || revision !== current || !diagram.figure.isConnected;
      diagram.figure.dataset.state = "loading";
      diagram.status.textContent = "Rendering diagram…";
      diagram.status.hidden = false;

      // Serialize initialize + render, not just render, because config is global.
      renderQueue = renderQueue.then(async () => {
        let staging: HTMLDivElement | undefined;
        try {
          if (stale()) return;
          if (diagram.source.length > MAX_SOURCE_LENGTH)
            throw new Error("Diagram too large");
          const { default: mermaid } = await import("mermaid");
          await document.fonts.ready;
          if (stale()) return;
          const config = diagramConfig();
          mermaid.initialize({
            ...config,
            secure: [
              ...(mermaid.mermaidAPI.defaultConfig.secure ?? []),
              ...Object.keys(config),
              "themeCSS",
              "dompurifyConfig",
            ],
          });
          staging = document.createElement("div");
          staging.className = "mermaid-staging";
          staging.setAttribute("aria-hidden", "true");
          staging.inert = true;
          document.body.append(staging);
          const { svg } = await mermaid.render(
            `mermaid-svg-${nextId++}`,
            diagram.source,
            staging,
          );
          if (stale()) return;
          diagram.canvas.innerHTML = svg;
          const image = diagram.canvas.querySelector("svg");
          if (!image) throw new Error("Missing diagram");
          image.setAttribute("role", "img");
          if (
            !image.hasAttribute("aria-label") &&
            !image.hasAttribute("aria-labelledby")
          ) {
            image.setAttribute("aria-label", diagram.label);
          }
          // Preserve readable text; wide diagrams scroll instead of shrinking.
          if (image.viewBox.baseVal.width > 0) {
            image.style.width = `${image.viewBox.baseVal.width}px`;
          }
          image.style.maxWidth = "none";
          image.style.height = "auto";
          diagram.viewport.hidden = false;
          diagram.status.hidden = true;
          diagram.figure.dataset.state = "ready";
        } catch {
          if (stale()) return;
          diagram.viewport.hidden = true;
          diagram.status.textContent =
            "This diagram could not render. Its Mermaid source is available below.";
          diagram.details.open = true;
          diagram.figure.dataset.state = "error";
        } finally {
          staging?.remove();
        }
      });
    }
  }

  render();
  const observer = new MutationObserver(() => {
    const next = document.documentElement.classList.contains("dark");
    if (next === dark) return;
    dark = next;
    render();
  });
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });

  return () => {
    disposed = true;
    observer.disconnect();
    diagrams.forEach(({ figure, pre }) => figure.replaceWith(pre));
  };
}
