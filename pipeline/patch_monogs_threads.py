"""Patch MonoGS slam.py to run the backend as a thread instead of a process.

WSL2's CUDA driver does not support CUDA IPC, so passing CUDA tensors between
processes fails with "CUDA error: invalid resource handle". Running the backend
in a thread of the same process avoids IPC entirely. The GUI (separate process)
is unchanged; use --eval (no GUI) on WSL.

Usage:  python patch_monogs_threads.py ~/MonoGS/slam.py   (idempotent)
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
if "MONOGS_THREADS" in src:
    print("already patched")
    sys.exit(0)

edits = [
    ("import torch.multiprocessing as mp\n",
     "import torch.multiprocessing as mp\nimport queue\nimport threading\n"
     "# WSL2 has no CUDA IPC: run the backend as a thread (set MONOGS_THREADS=0 to disable)\n"
     "USE_THREADS = os.environ.get(\"MONOGS_THREADS\", \"1\") == \"1\"\n"),
    ("        frontend_queue = mp.Queue()\n        backend_queue = mp.Queue()\n",
     "        Q = queue.Queue if USE_THREADS else mp.Queue\n"
     "        frontend_queue = Q()\n        backend_queue = Q()\n"),
    ("        backend_process = mp.Process(target=self.backend.run)\n",
     "        if USE_THREADS:\n"
     "            backend_process = threading.Thread(target=self.backend.run, daemon=True)\n"
     "        else:\n"
     "            backend_process = mp.Process(target=self.backend.run)\n"),
]
for old, new in edits:
    assert src.count(old) == 1, f"anchor not found exactly once: {old!r}"
    src = src.replace(old, new)
assert "import os" in src, "slam.py must import os before the patch block"
open(path, "w", encoding="utf-8").write(src)
print("patched")
