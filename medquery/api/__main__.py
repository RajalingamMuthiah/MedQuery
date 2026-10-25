"""Run the API locally: uv run python -m medquery.api (http://127.0.0.1:8000)."""

import os

import uvicorn

from medquery.api.app import create_app


def main() -> None:
    uvicorn.run(
        create_app(),
        host=os.environ.get("MEDQUERY_HOST", "127.0.0.1"),
        port=int(os.environ.get("MEDQUERY_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
