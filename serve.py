#!/usr/bin/env python3
"""Local dev server for the triple pendulum viewer.

Serves docs/ as the web root, with /data/ mapped to the project-root data/ directory.
This matches the production layout where the viewer fetches data from a relative data/ path.

Usage:
    python3 serve.py [--port 8000]
"""

import argparse
import os
import http.server
import functools


class DualRootHandler(http.server.SimpleHTTPRequestHandler):
    """Serves docs/ by default, but routes /data/ requests to project-root data/."""

    def __init__(self, *args, docs_dir, data_dir, **kwargs):
        self.docs_dir = docs_dir
        self.data_dir = data_dir
        # SimpleHTTPRequestHandler uses 'directory' kwarg for its root
        super().__init__(*args, directory=docs_dir, **kwargs)

    def translate_path(self, path):
        """Route /data/* to project-root data/, everything else to docs/."""
        # Normalize path
        path = path.split("?", 1)[0].split("#", 1)[0]

        if path.startswith("/data/") or path == "/data":
            # Strip /data/ prefix and serve from project-root data/
            rel = path[len("/data/"):]
            return os.path.join(self.data_dir, rel)

        # Default: serve from docs/
        return super().translate_path(path)


def main():
    parser = argparse.ArgumentParser(description="Triple pendulum local dev server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(project_root, "docs")
    data_dir = os.path.join(project_root, "data")

    handler = functools.partial(
        DualRootHandler, docs_dir=docs_dir, data_dir=data_dir
    )

    with http.server.HTTPServer(("", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"Serving viewer at {url}")
        print(f"  docs/ -> {docs_dir}")
        print(f"  data/ -> {data_dir}")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
