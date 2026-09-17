import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.evaluation.evaluate import run_evaluation


if __name__ == "__main__":
    import json
    print(json.dumps(run_evaluation(), indent=2))
