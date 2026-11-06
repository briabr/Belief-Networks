# %%
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import os

bigfs = 9
smallfs = 7
plt.rcParams.update({"font.size": bigfs})
plt.rcParams.update({"axes.titlesize": bigfs})
plt.rcParams.update({"axes.labelsize": bigfs})
plt.rcParams.update({"legend.fontsize": smallfs})
plt.rcParams.update({"xtick.labelsize": smallfs})
plt.rcParams.update({"ytick.labelsize": smallfs})

beforeRange = range(91, 101)
duringRange = range(141, 151)
afterRange = range(191, 201)
# %%
belief_dimensions = [int(a) for a in range(10)]
edgelist = list(combinations(belief_dimensions, 2))
edges_columns = [f"w{a}{b}" for a, b in edgelist]

namesTex = {
    (0.2, 0.0, False): r"static ($\omega_0=0.2$)",
    (0.2, 1.0, False): r"adaptive ($\omega_0=0.2$)",
}
# %%
seeds = [0, 1, 2, 3, 4,]
res = []
beta = 3.00
mean_absedges = []
param_combis = [
    (0.2, 0.0, False),   # low ω₀, no internal adaptation
    (0.2, 1.0, False),   # low ω₀, internal adaptation
]
for init_w, eps, fixedBNat100 in param_combis:

        for seed in seeds:

            df = pd.read_csv(
                f"simOut_n1_nopressure/detailed/sim_init_w{init_w:.2f}_beta{beta:.2f}_eps{eps:.2f}_seed{seed}_detailed.csv"
            )
            print("FILE:", f"simOut_n1_nopressure/detailed/sim_init_w{init_w:.2f}_beta{beta:.2f}_eps{eps:.2f}_seed{seed}_detailed.csv"
                        )
            print("NUMBER OF AGENTS:", df.loc[df.t == 100, "id"].nunique())
            print("EDGE SHAPE:", df.loc[df.t == 100, edges_columns].shape)
            print(df.loc[df.t == 100, edges_columns].shape)
            before_focal = df.loc[df.t.isin(beforeRange), "0"].mean()
            during_focal = df.loc[df.t.isin(duringRange), "0"].mean()
            after_focal = df.loc[df.t.isin(afterRange), "0"].mean()

            before_focal_negative = before_focal < 0
            res.append(
                [
                    init_w,
                    eps,
                    fixedBNat100,
                    namesTex[(init_w, eps, fixedBNat100)],
                    before_focal,
                    during_focal,
                    after_focal,
                    before_focal_negative,
                ]
            )

            mean_absedges.append(
                [
                    init_w,
                    eps,
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
        "fixedBNat100",
        "name",
        "before_focal",
        "during_focal",
        "after_focal",
        "before_focal_negative",
        # "std_focal",
    ],
)
# %%
sns.barplot(
    data=pd.DataFrame(
        mean_absedges,
        columns=[
            "init_w",
            "eps",
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
    y="mean_abs_edge",
    palette="plasma",
)
# %%
# res.groupby("name").std_focal.mean(),
# res.groupby("name").std_focal.std()
# # %%

# print(res.groupby("name").std_focal.mean())
# print(res.groupby("name").std_focal.std())
print(
    res.groupby(["name"])[
    ["before_focal", "during_focal", "after_focal"]
].mean()
)

print(
    pd.DataFrame(
        mean_absedges,
        columns=["init_w", "eps", "fixedBNat100", "name", "absOm_tot", "Om_tot"],
    ).groupby(["name"]).absOm_tot.mean()
)
os.makedirs("2026-11_figs", exist_ok=True)
plt.savefig("2026-11_figs/individual_nopressure_analysis.png", dpi=600)
print("saved 2026-11_figs/individual_nopressure_analysis.png")