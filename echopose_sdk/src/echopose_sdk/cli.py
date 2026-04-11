from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validation import validate_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="echopose-sdk utilities")
    parser.add_argument("--bundle", type=Path, help="Path to CSI bundle JSON")
    args = parser.parse_args()

    if not args.bundle:
        parser.print_help()
        return

    payload = json.loads(args.bundle.read_text(encoding="utf-8"))
    ok, msg = validate_bundle(payload)
    print(json.dumps({"valid": ok, "reason": msg}))


if __name__ == "__main__":
    main()
