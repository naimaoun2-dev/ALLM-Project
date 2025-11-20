import os
from pathlib import Path
import uncompyle6

# Paths
pycache_folder = Path(r"C:\Users\naim-\Desktop\Masters LAU\Fall 2025\LLM\FinalProject\__pycache__")
out_folder = Path(r"C:\Users\naim-\Desktop\Masters LAU\Fall 2025\LLM\FinalProject\recovered_source")
out_folder.mkdir(exist_ok=True)

# Decompile all .pyc files
for pyc_file in pycache_folder.glob("*.pyc"):
    py_file = out_folder / (pyc_file.stem + ".py")
    try:
        with open(py_file, "w", encoding="utf-8") as f:
            uncompyle6.decompile_file(str(pyc_file), f)
        print(f"✅ Decompiled {pyc_file.name} → {py_file.name}")
    except Exception as e:
        print(f"❌ Failed to decompile {pyc_file.name}: {e}")
