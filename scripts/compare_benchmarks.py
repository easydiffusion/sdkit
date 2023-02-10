import sys

import pandas as pd

stable = pd.read_csv(sys.argv[1])
beta = pd.read_csv(sys.argv[2])

del stable["vram_usage"], stable["ram_usage"]
del beta["vram_usage"], beta["ram_usage"]

del (
    stable["render_test"],
    stable["vram_tp90"],
    stable["vram_spike_test"],
    stable["overall_status"],
)
del (
    beta["render_test"],
    beta["vram_tp90"],
    beta["vram_spike_test"],
    beta["overall_status"],
)

bad = stable["max_vram (GB)"] < beta["max_vram (GB)"]  # & (stable['model_filename'] == 'sd-v1-4.ckpt')
good = stable["max_vram (GB)"] > beta["max_vram (GB)"]  # & (stable['model_filename'] == 'sd-v1-4.ckpt')

print("bad results", bad)
print("good results", good)
