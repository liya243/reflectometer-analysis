import re
from pathlib import Path


def matlab_safe_stem(text):
    safe = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not safe or not safe[0].isalpha():
        safe = f"fig_{safe}"
    return safe


def cleanup_outputs_for_dataset(output_dir, dataset_stem):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    safe_stem = matlab_safe_stem(dataset_stem)
    patterns = [
        f"{dataset_stem}*",
        f"open_{safe_stem}*",
    ]
    removed = []
    seen = set()
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            path.unlink()
            removed.append(path)
    return removed
