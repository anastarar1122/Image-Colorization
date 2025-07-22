from pathlib import Path
import shutil

# ---- CONFIG ----
source_dir = Path(r"F:\.vscode\Projects\p4")
destination_dir = Path(r"F:\.vscode\Projects\p4_clone")

# ---- CLEAN START (skip deletion if folder is in use) ----
if not destination_dir.exists():
    destination_dir.mkdir(parents=True, exist_ok=True)
else:
    print(f"❌ Destination folder '{destination_dir}' already exists, skipping deletion.")

# ---- WALK AND CLONE ----
for path in source_dir.rglob('*'):
    relative_path = path.relative_to(source_dir)
    dest_path = destination_dir / relative_path

    if path.is_dir():
        dest_path.mkdir(parents=True, exist_ok=True)
    elif path.is_file():
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.touch(exist_ok=True)

print(f"✅ Structure cloned from:\n{source_dir}\n→ To:\n{destination_dir}")
