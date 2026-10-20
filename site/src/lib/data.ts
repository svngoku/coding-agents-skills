import { useEffect, useState } from "react";
import type { Gallery } from "./types";

type GalleryState =
  | { status: "loading"; data: null; error: null }
  | { status: "error"; data: null; error: string }
  | { status: "ready"; data: Gallery; error: null };

const GALLERY_URL = `${import.meta.env.BASE_URL}gallery.json`;

export function useGallery(): GalleryState {
  const [state, setState] = useState<GalleryState>({
    status: "loading",
    data: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    fetch(GALLERY_URL, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`The gallery payload returned HTTP ${response.status}.`);
        }
        return response.json() as Promise<Gallery>;
      })
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled || controller.signal.aborted) return;
        const message =
          error instanceof Error ? error.message : "The gallery payload could not be loaded.";
        setState({ status: "error", data: null, error: message });
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  return state;
}
