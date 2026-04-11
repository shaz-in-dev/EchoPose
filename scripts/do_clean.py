import os
import shutil
from pathlib import Path

root = Path(r"C:\Users\Admin\wifi vision")

# 1. Clean Scripts folder
scripts_dir = root / "scripts"
if scripts_dir.exists():
    for f in scripts_dir.iterdir():
        if f.is_file() and f.name != "mock_esp32_mesh.py":
            print(f"Deleting {f}")
            f.unlink()

# 2. Clean __pycache__
for d in root.rglob("__pycache__"):
    if d.is_dir():
        print(f"Deleting {d}")
        shutil.rmtree(d)

print("Cleanup complete.")
