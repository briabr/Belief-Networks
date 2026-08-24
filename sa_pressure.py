# %%
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import netCDF4
import json
import xarray as xr
import matplotlib as mpl
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist

import matplotlib.path as mpath
import numpy as np

plt.rcParams.update({"font.size": 10})
bigfs = 9
smallfs = 7
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})


# %%
eps = 1.0
response_cols = [
    "persistPos",
    "nonpersistPos",
    "compliant",
    "resilient",
    "resistant",
    "latecompliant",
]
belief_dimensions = [int(a) for a in range(10)]
belief_columns = [str(int(a)) for a in belief_dimensions]
edgelist = list(combinations(belief_dimensions, 2))
edges_columns = [f"w{a}{b}" for a, b in edgelist]
cmap = dict(
    zip(
        response_cols + ["NA"],
        ["#4CAF50", "#AED581", "#2196F3", "#9C27B0", "#F44336", "#90CAF9", "#9E9E9E"],
    )
)
s_exts = [4]
names = {
    (0.2, 0.0, False): r"static ($\omega_0=0.2$)",
}
experiments = {
    "beta": r"attention to dissonance $\beta$",
    "M": r"nr of beliefs $M$",
    "tau": r"nr of past belief changes" + "\n" + r"kept in memory $\tau$",
}

res = []
# M N initw eps mu beta p s responsFreq

mean_absedges = []
pressure = 4
condition_pairs = [(0.2, 0.0)]

param_combis = (
    [
        ["base", 10, w, eps, 3.0, 1, pressure]
        for w, eps in condition_pairs
    ]
    + [
        ["M", M, w, eps, 3.0, 1, pressure]
        for w, eps in condition_pairs
        for M in [5, 15]
    ]
    + [
        ["beta", 10, w, eps, beta, 1, pressure]
        for w, eps in condition_pairs
        for beta in [1.5, 6.0]
    ]
    + [
        ["tau", 10, w, eps, 3.0, tau, pressure]
        for w, eps in condition_pairs
        for tau in [2.0, 10.0]
    ]
)
fixedBNatt100 = False

for exp, M, init_w, eps, beta,tau, s in param_combis:
    print(exp, end=",")
    for seed in [0]:
        addon = (
            f"_M{M:.2f}"
            if exp == "M"
                else (f"_tau{tau:.2f}" if exp == "tau" else "")
            )
        
        df = pd.read_csv(
    f"simOut/sim_init_w{init_w:.2f}_beta{beta:.2f}_eps{eps:.2f}"
    f"{'_fixedBNat100' if fixedBNatt100 else ''}"
    f"_ext_strength{s}_seed{seed}{addon}.csv"
)
        res.append(
            [
                exp,
                M,
                init_w,
                eps,
                beta,
                tau,
                s,
                names[(init_w, eps, fixedBNatt100)],
            ]
            + df.loc[df.t == 95.5][response_cols].sum(axis=0)[response_cols].to_list()
        )
# %%
res = pd.DataFrame(
    res,
    columns=[
        "exp",
        "M",
        "init_w",
        "eps",
        "beta",
        "tau",
        "s_ext",
        "name",
    ]
    + response_cols,
)
# %%
relres = res[["compliant", "resilient", "resistant", "latecompliant"]].div(
    res[["compliant", "resilient", "resistant", "latecompliant"]].sum(axis=1), axis=0
)
for c in ["exp", "M", "init_w", "eps", "beta", "tau", "s_ext", "name"]:
    relres[c] = res[c]
relres["tau"] = (relres["tau"]).astype(int)
# %%

relres = relres.loc[relres.s_ext > 0]
fig, axs = plt.subplots(3, 1, sharex=False, sharey=True, figsize=(18 / 2.54, 12 / 2.54))
T = 200
for n, name in enumerate(["M","beta", "tau"]):
    for nn, (init_w, eps) in enumerate(condition_pairs):
        ax = axs[n]
        subset = relres.query(
            f"exp=='{name}' and eps == {eps} and init_w=={init_w}"
        )
        subset = subset[["compliant", "resilient", "resistant", name]].melt(
            id_vars=name, value_name="normalized_count", var_name="response"
        )
        base = relres.query(
            f"exp=='base' and eps == {eps} and init_w=={init_w}"
        )
        base = base[["compliant", "resilient", "resistant", name]].melt(
            id_vars=name, value_name="normalized_count", var_name="response"
        )
        subset = pd.concat([subset, base])
        subset = subset.reset_index()
        # ax = sns.boxplot(subset, ax=ax, x=name, hue="response", y="normalized_count", palette=cmap, hue_order=["compliant", "resilient", "resistant"], legend=False, fliersize=0, fill=True, linewidth=0., whis=0)
        ax = sns.stripplot(
            subset,
            ax=ax,
            x=name,
            hue="response",
            y="normalized_count",
            jitter=True,
            palette=cmap,
            hue_order=["compliant", "resilient", "resistant"],
            legend=False,
            size=2,
            alpha=0.4,
            dodge=True,
        )

        for coll in ax.collections:
            coll.set_clip_on(False)
        avgs = (
            subset.groupby(["response", name])["normalized_count"]
            .median()
            .reset_index()
        )
        ax = sns.stripplot(
            avgs,
            ax=ax,
            x=name,
            hue="response",
            y="normalized_count",
            jitter=True,
            palette=cmap,
            hue_order=["compliant", "resilient", "resistant"],
            legend=False,
            size=4,
            alpha=0.8,
            dodge=True,
            marker="s",
        )
        for coll in ax.collections:
            coll.set_clip_on(False)
        if n == 2:
            ax.set_ylabel("response frequency", fontsize=bigfs, va="center")
        else:
            ax.set_ylabel("")
        if n == 0:
            ax.set_title(
                names[(init_w, eps, fixedBNatt100)],
                fontsize=bigfs,
                x=0.5,
                y=0.99,
                ha="center",
            )
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=bigfs)

        ax.text(
                1.02,
                0.1,
                experiments[name],
                va="center",
                ha="left",
                transform=ax.transAxes,
                fontdict={"weight": "bold"},
            )
        ax.set_xlabel("")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([-0.0, 0.5, 1.0])
        ax.set_yticklabels(rf"${int(x*100)}\,\%$" for x in [0, 0.5, 1.0])

import string

for n, ax in enumerate(axs.flatten()):
    ax.text(
        0.0 if n == 0 else 0,
        1.01,
        string.ascii_uppercase[n],
        fontsize=12,
        fontdict={"weight": "bold"},
        va="bottom",
        ha="left",
        transform=ax.transAxes,
    )

filename = "2026-04_figs/fig-sa_ofat.png"
print(filename)
fig.subplots_adjust(
    hspace=0.6, wspace=0.05, top=0.95, left=0.08, right=0.76, bottom=0.06
)
plt.savefig(filename)

# %%
