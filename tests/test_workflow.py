import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import FEATURES

ROOT = Path(__file__).resolve().parents[1]


class KeplerWorkflowTests(unittest.TestCase):
    def test_local_csv_cli_handles_missing_features_and_writes_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(42)
            target = np.repeat([0, 1], 60)
            values = rng.normal(size=(len(target), len(FEATURES))) + target[:, None]
            frame = pd.DataFrame(values, columns=FEATURES)
            frame["kepid"] = np.repeat(np.arange(60), 2)
            frame["koi_disposition"] = np.where(target == 1, "CONFIRMED", "FALSE POSITIVE")
            frame.loc[::7, FEATURES[0]] = np.nan
            source = root / "kepler.csv"
            frame.to_csv(source, index=False)
            output = root / "results"
            result = subprocess.run(
                [sys.executable, "evaluate.py", "--data", str(source), "--output", str(output), "--cluster"],
                cwd=ROOT, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["train_rows"] + metrics["test_rows"], len(frame))
            self.assertEqual(metrics["train_stars"] + metrics["test_stars"], 60)
            self.assertEqual(len(metrics["holdout"]), 3)
            self.assertEqual(len(metrics["dataset_sha256"]), 64)
            self.assertTrue((output / "roc.png").is_file())

            notebook_data = root / "data" / "kepler-koi.csv"
            notebook_data.parent.mkdir()
            frame.to_csv(notebook_data, index=False)
            notebook = subprocess.run(
                [sys.executable, "-c", """
import sys
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

path = Path(sys.argv[1])
sys.path.insert(0, str(path.parent))
book = nbformat.read(path, as_version=4)
nbformat.validate(book)
shell = InteractiveShell.instance()
for cell in book.cells:
    if cell.cell_type == "code":
        shell.run_cell(cell.source).raise_error()
""", str(ROOT / "Kepler_Project.ipynb")],
                cwd=root, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(notebook.returncode, 0, notebook.stderr)
            self.assertTrue((root / "outputs" / "roc.png").is_file())
