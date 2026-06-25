"""Print common platform directories to locate uploaded datasets."""
from pathlib import Path


def main() -> None:
    candidates = [
        ".",
        "data",
        "datasets",
        "/home/jovyan",
        "/home/jovyan/work",
        "/home/jovyan/work/datasets",
        "/mnt/data",
    ]
    for candidate in candidates:
        path = Path(candidate)
        print(f"\n== {candidate} ==")
        print(f"exists: {path.exists()}")
        if not path.exists():
            continue
        print(f"resolved: {path.resolve()}")
        try:
            for item in list(path.iterdir())[:30]:
                suffix = "/" if item.is_dir() else ""
                print(f"  {item.name}{suffix}")
        except PermissionError as exc:
            print(f"  permission denied: {exc}")


if __name__ == "__main__":
    main()

