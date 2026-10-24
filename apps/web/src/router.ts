import { useSyncExternalStore } from "react";

export type Route = { name: "upload" } | { name: "analysis"; id: string } | { name: "notFound" };

const NAVIGATE = "app:navigate";

export function parseRoute(pathname: string): Route {
  if (pathname === "/" || pathname === "") return { name: "upload" };
  const match = /^\/analyses\/([0-9a-f]{32})\/?$/.exec(pathname);
  if (match?.[1]) return { name: "analysis", id: match[1] };
  return { name: "notFound" };
}

function subscribe(callback: () => void): () => void {
  window.addEventListener("popstate", callback);
  window.addEventListener(NAVIGATE, callback);
  return () => {
    window.removeEventListener("popstate", callback);
    window.removeEventListener(NAVIGATE, callback);
  };
}

export function usePathname(): string {
  return useSyncExternalStore(subscribe, () => window.location.pathname);
}

export function navigate(path: string, { replace = false } = {}): void {
  if (replace) window.history.replaceState(null, "", path);
  else window.history.pushState(null, "", path);
  window.dispatchEvent(new Event(NAVIGATE));
  window.scrollTo?.(0, 0);
}
