import sys
import pandas

# import seaborn as sns
import matplotlib.pyplot as plt

stable = pandas.read_csv(sys.argv[1])
beta = pandas.read_csv(sys.argv[2])

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

bad = (
    stable["max_vram (GB)"] < beta["max_vram (GB)"]
)  # & (stable['model_filename'] == 'sd-v1-4.ckpt')
good = (
    stable["max_vram (GB)"] > beta["max_vram (GB)"]
)  # & (stable['model_filename'] == 'sd-v1-4.ckpt')

print("bad results", bad)
print("good results", good)

# combined = pandas.concat([stable, beta], axis=0, ignore_index=False)
# combined['version'] = (len(stable)*('v2.4',) + len(beta)*('v2.5',))
# combined.reset_index(inplace=True)

# combined['index'] = combined['model_filename'].apply(lambda x: x.replace('.ckpt', '').replace('.safetensors', '')) + ',' + combined['vram_usage_level'] + ',' + combined['image_size']

# sns.catplot(x='index', y='max_vram (GB)', hue='version', kind='bar', data=combined)
# plt.xticks(rotation=90, fontsize=6)
# plt.show()
