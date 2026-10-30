# %%
# from matplotlib.pylab import beta
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4
import json
import xarray as xr
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist
import matplotlib.path as mpath
import glob
import os

plt.rcParams.update({"font.size": 10})
bigfs = 9
smallfs = 7
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})

T = 200
beforeRange = range(91, 101)
duringRange = range(141, 151)
afterRange = range(191, 201)
# %%
belief_dimensions = [int(a) for a in range(10)]
belief_columns = [str(int(a)) for a in belief_dimensions]
edgelist = list(combinations(belief_dimensions, 2))
edges_columns = [f"w{a}{b}" for a, b in edgelist]

names = {
    (0.2, 0.0, False): "staticlow",
    (0.8, 0.0, False): "statichigh",
    (0.2, 1.0, False): "adaptivelow",
    (0.8, 1.0, False): "adaptivehigh",
}

namesTex = {
    (0.2, 0.0, False): r"static ($\omega_0=0.2$)",
    (0.8, 0.0, False): r"static ($\omega_0=0.8$)",
    (0.2, 1.0, False): r"adaptive ($\omega_0=0.2$)",
    (0.8, 1.0, False): r"adaptive ($\omega_0=0.8$)",
}
# %%
s_exts = [0, 1, 2, 4, 8, 16]
seeds = [0, 1, 2, 3, 4,]
res = []
beta = 3.00
mean_absedges = []
param_combis = [
    (0.2, 0.0, False),   # low ω₀, no internal adaptation
    (0.2, 1.0, False),   # low ω₀, internal adaptation
]
for init_w, eps, fixedBNat100 in param_combis:
    for s in s_exts:
        for seed in seeds:
            df = pd.read_csv(
    f"simOut/detailed/sim_init_w{init_w:.2f}_beta{beta:.2f}_eps{eps:.2f}_ext_strength{s}_seed{seed}_detailed.csv"
)
            W = df.loc[df.t == 100, edges_columns].values
            dists = pdist(W, metric="cityblock")
            groupishness = 0 if eps == 0 else dists.std() / dists.mean()
            # std_focal = df.loc[df.t == 100, "0"].std()
            before_focal = df.loc[df.t.isin(beforeRange), "0"].mean()
            during_focal = df.loc[df.t.isin(duringRange), "0"].mean()
            after_focal = df.loc[df.t.isin(afterRange), "0"].mean()
            nr_negs = sum(
                df.loc[df.t.isin(range(91, 101)), ["0", "id", "t"]]
                .pivot_table(index="id", values="0", columns="t")
                .mean(axis=1)
                < 0
            )
            res.append(
    [
        init_w,
        eps,
        s,
        fixedBNat100,
        namesTex[(init_w, eps, fixedBNat100)],
        groupishness,
        # std_focal,
        before_focal,
        during_focal,
        after_focal,
        nr_negs,
    ]
)

            mean_absedges.append(
                [
                    init_w,
                    eps,
                    s,
                    fixedBNat100,
                    namesTex[(init_w, eps, fixedBNat100)],
                ]
                + [np.abs(df.loc[df.t == 100, edges_columns].values).mean()]
                + [df.loc[df.t == 100, edges_columns].values.mean()]
            )

res = pd.DataFrame(
    res,
    columns=[
        "init_w",
        "eps",
        "s_ext",
        "fixedBNat100",
        "name",
        "groupishness",
        "before_focal",
        "during_focal",
        "after_focal",
        "nr_negs",
        # "std_focal",
    ],
)
# %%
sns.barplot(
    pd.DataFrame(
        mean_absedges,
        columns=[
            "init_w",
            "eps",
            "s_ext",
            "fixedBNat100",
            "name",
            "mean_abs_edge",
            "mean_edge",
        ],
    )
    .groupby(["name"])
    .mean()
    .reset_index(),
    hue="name",
    x="name",
    y="mean_abs_edge",
    palette="plasma",
)
# %%
# res.groupby("name").std_focal.mean(),
# res.groupby("name").std_focal.std()
# # %%
pd.DataFrame(
    mean_absedges,
    columns=["init_w", "eps", "s_ext", "fixedBNat100", "name", "absOm_tot", "Om_tot"],
).groupby("name").absOm_tot.mean()

# print(res.groupby("name").std_focal.mean())
# print(res.groupby("name").std_focal.std())
print(
    res.groupby("name")[["before_focal", "during_focal", "after_focal"]].mean().to_string()
)

print(
    pd.DataFrame(
        mean_absedges,
        columns=["init_w", "eps", "s_ext", "fixedBNat100", "name", "absOm_tot", "Om_tot"],
    ).groupby("name").absOm_tot.mean()
)

plt.savefig("2026-04_figs/nopressure_analysis.png", dpi=600)
print("saved 2026-04_figs/nopressure_analysis.png")