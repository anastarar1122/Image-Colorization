from pathlib import Path

# Define base directory
base_dir = Path(r"F:\.vscode\Projects\Image_Colorization_Project")

# Folders you want to exclude (case-sensitive)
excluded_dirs = {'__pycache__', '.git', '.github', 'wheels', 'data', 'imgcolor_env'}

# Recursively walk through all files, skipping excluded directories
for path in base_dir.rglob('*'):
    # Skip directories (not files) in excluded_dirs
    if path.is_dir() and any(excluded_dir in path.parts for excluded_dir in excluded_dirs):
        continue

    # Only print files
    if path.is_file():
        print(path.relative_to(base_dir))
