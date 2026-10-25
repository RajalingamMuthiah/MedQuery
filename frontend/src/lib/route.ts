import { useEffect, useState } from "react";

export type Route = "ask" | "sources" | "about";

export function parseRoute(hash: string): Route {
  const name = hash.replace(/^#\/?/, "");
  return name === "sources" || name === "about" ? name : "ask";
}

/** The current page from the URL hash (#/, #/sources, #/about); links and Back just work. */
export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
