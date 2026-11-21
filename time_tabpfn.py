"""
Time real TabPFN for 500 / 1000 / 2000 context rows (predicting one 200-row block).
Also confirms the licence token works. Run on Colab with a GPU:  python time_tabpfn.py
"""
import os
import sys
import time
from common.contract import FEATURE_COLS, N_BLOCKS
from data.streams import make_stream

print("TABPFN_TOKEN set:", bool(os.environ.get("TABPFN_TOKEN")),
      "| HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))

import torch
if not torch.cuda.is_available():
    sys.exit("No GPU found. Run this on Colab (Runtime > Change runtime type > T4 GPU).")
print("GPU:", torch.cuda.get_device_name(0))

from methods.tabpfn_wrapper import TabPFNModel

try:
    model = TabPFNModel(seed=0)
    df, _ = make_stream("none", 0)
    X, y = df[FEATURE_COLS].to_numpy(), df["y"].to_numpy()
    model.fit(X[:100], y[:100]).predict(X[100:110])     # warm-up: loads the weights
except Exception as e:
    sys.exit(f"TabPFN failed to load: {e}\n"
             "Most likely a missing/invalid token: set TABPFN_TOKEN (ux.priorlabs.ai) "
             "or HF_TOKEN (accept the model terms on huggingface.co first).")

test_X = X[-200:]
for n in (500, 1000, 2000):
    t0 = time.time()
    for _ in range(3):
        model.fit(X[:n], y[:n]).predict(test_X)
    sec = (time.time() - t0) / 3
    print(f"{n:5d} context rows: {sec:.2f} s per block  (~{sec * (N_BLOCKS - 1) / 60:.1f} min per vanilla run)")
