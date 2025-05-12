import matplotlib.pyplot as plt

import seaborn as sns
import csv
import pandas as pd

import shutil

SHARE_DIR = "/imec/other/dtpatha/patel23/share"


categories = [
    "Q_K_V",
    "Q_mul_K",
    "A_mul_V",
    "Wo_proj",
    "W1_proj",
    "W2_proj",
    "Softmax",
    "LayerNorm_MHA",
    "LayerNorm_FFN",
    "GeLU",
    "AllReduce_MHA",
    "AllReduce_FFN",
]
col_names = ["area", "latency"] + categories

colors_matmul = sns.color_palette("flare_r", 6)
colors_normalization = sns.color_palette("summer", 3)
colors_gelu = sns.color_palette("pink", 1)
colors_allreduce = sns.color_palette("Blues_r", 2)
colors = colors_matmul + colors_normalization + colors_gelu + colors_allreduce

batch_size = 8

simRuns = []
with open("simRuns.txt", "r") as file:
    simRuns = file.readlines()


initList = []
ar_list = []
for cSim in simRuns:
    cSim = cSim.replace("\n", "")
    # print(cSim)
    _tmpSplitStr = cSim.split("_")
    results_init = pd.read_csv(
        f"results/{cSim}_init.csv",
        header=None,
        names=col_names,
        index_col=0,
    )
    results_init.index.astype(float)
    results_ar = pd.read_csv(
        f"results/{cSim}_ar.csv",
        header=None,
        names=col_names,
        index_col=0,
    )
    results_ar.index.astype(float)
    # print(results_init)
    # print(results_init.index.tolist())

    x = 0

    # x_labels = [i * 400 for i in [1, 2, 3, 4, 5, 6, 7, 8]]
    x_labels = results_init.index.tolist()
    for row_index in x_labels:
        x = x + 1
        try:
            values = results_init.loc[row_index].tolist()
            # print(results_init.loc[row_index])
        except:
            values = results_init.iloc[x - 1].tolist()
            # print(results_init.iloc[x - 1])

        bottom = 0
        # for i, (category, value) in enumerate(zip(categories, values[2:])):
        cData = {
            "Sim": cSim,
            "Bug": str(x),
            "Memory": _tmpSplitStr[1],
            "Mem per chiplet": _tmpSplitStr[2],
            "Mem BW": row_index,
            "Interconnect": _tmpSplitStr[4],
        }
        for _key, _value in zip(categories, values[2:]):
            cData[_key] = _value

        initList.append(cData)

    # break
    # Set the title, legend, and display the graph
    # plt.title(
    #     "Prefilling Latency per Layer"
    # )

    x = 0
    x_labels = results_ar.index.tolist()
    for row_index in x_labels:
        x = x + 1
        try:
            values = results_ar.loc[row_index].tolist()
            # print(results_init.loc[row_index])
        except:
            values = results_ar.iloc[x - 1].tolist()
            # print(results_init.iloc[x - 1])
        # values = results_ar.loc[row_index].tolist()
        bottom = 0
        # for i, (category, value) in enumerate(zip(categories, values[2:])):
        cData = {
            "Sim": cSim,
            "Bug": str(x),
            "Memory": _tmpSplitStr[1],
            "Mem per chiplet": _tmpSplitStr[2],
            "Mem BW": row_index,
            "Interconnect": _tmpSplitStr[4],
        }
        for _key, _value in zip(categories, values[2:]):
            cData[_key] = _value

        ar_list.append(cData)
        value = value * 1e3

    # Set the title, legend, and display the graph
    # plt.title(
    #     "Generation Latency per Layer per Token"
    # )
with open(f"prefill.csv", "w", newline="") as csvfile:
    fieldnames = initList[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(initList)

destFile = f"{SHARE_DIR}/llmcompass/prefill.csv"
shutil.copy(f"prefill.csv", destFile)


with open(f"generation.csv", "w", newline="") as csvfile:
    fieldnames = ar_list[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ar_list)

destFile = f"{SHARE_DIR}/llmcompass/generation.csv"
shutil.copy(f"generation.csv", destFile)
