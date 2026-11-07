"""E5: neutral (non-licensing) decomposition instruction on the perturbed biographies (greedy)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "rerun"))
import r25_decompose
from phase0.decomposition.decompose import DECOMPOSE_PROMPT
r25_decompose.ALL["perturbed_neutral"] = ("perturbed_bio", DECOMPOSE_PROMPT)
sys.argv = ["r25", "--conditions", "perturbed_neutral"]
r25_decompose.main()
