"""Launch an independently installed upstream VERA-MH checkout."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--python", default=sys.executable, help="Python executable in the upstream environment")
    p.add_argument("--entry", choices=["pipeline", "generate", "judge"], default="pipeline")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Upstream arguments after --")
    a = p.parse_args()
    repo = a.repo.resolve()
    script = repo / {"pipeline": "run_pipeline.py", "generate": "generate.py", "judge": "judge.py"}[a.entry]
    if not script.is_file():
        p.error(f"Upstream entry point missing: {script}; clone SpringCare/VERA-MH first")
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    executable = str(Path(a.python).resolve()) if "/" in a.python else a.python
    subprocess.run([executable, str(script), *args], cwd=repo, check=True)


if __name__ == "__main__":
    main()
