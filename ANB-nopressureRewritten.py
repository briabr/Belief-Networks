# %%
"""
Adaptive Belief Networks Model
version 2026-04-15, Peter Steiglechner, steiglechner@csh.ac.at
"""
import numpy as np
import pandas as pd
import os
from itertools import combinations
from joblib import Parallel, delayed
import multiprocessing

# FIXED PARAMETERS
M = 10
focal = 0
n_agents = 100
tau = 1
T = 200 
detail = True
track_times = np.arange(T + 1)

belief_dimensions = list(range(0, M))
beliefupdate_order = belief_dimensions.copy()

belief_neighbours = {
    i: [nj for nj, j in enumerate(belief_dimensions) if j != i]
    for i in belief_dimensions
}

agentids = list(range(n_agents))

edge_list = list(combinations(belief_dimensions, 2))
edge_arr = np.array(edge_list)

edge_lookup = {c: nc for nc, c in enumerate(edge_list)}

adjacent_edge_ids = {
    dim: [
        edge_lookup[(dim, j)] if (dim, j) in edge_list else edge_lookup[(j, dim)]
        for j in belief_dimensions
        if j != dim
    ]
    for dim in belief_dimensions
}

belief_options = np.linspace(-1, 1, 21)
n_options = len(belief_options)


# PREPARE OUTPUT DATASET
meta_cols = ["t", "id"]
metric_cols = [
   
    "absOm_tot",
    "absOm_foc",
    "expI",
    "x_focal",
    "extr_nonfoc",
]
response_cols = []
columnIndex = len(meta_cols) + len(metric_cols) + len(response_cols)
beliefids = list(range(columnIndex, columnIndex + M))
edgeids = list(range(columnIndex + M, columnIndex + M + len(edge_list)))
delbeliefids = range(
    columnIndex + M + len(edge_list), columnIndex + M + len(edge_list) + tau * M
)
belief_cols = [str(b) for b in belief_dimensions]
edge_cols = [f"w{e[0]}{e[1]}" for e in edge_list]
columns = meta_cols + metric_cols + response_cols + belief_cols + edge_cols


def initialise_agents(init_w):
    edgeweights = [init_w] * len(edge_list)
    agent_list = []
    for id in range(n_agents):
        opinion_vector = np.random.choice(
            belief_options, size=M, p=np.ones(n_options) / n_options
        )
        agent_list.append(
            [0, id]
            + [np.nan] * (columnIndex - 2)
            + list(opinion_vector)
            + edgeweights
            + list(np.zeros(M)) * tau  # this will be used to store past belief changes
        )
    return np.array(agent_list)

# The core function that decides how a belief can change.
def glauber_fast(dim, agent, beta):
    old_belief = agent[beliefids][dim]

    adjacent_beliefs = agent[beliefids][belief_neighbours[dim]]
    adjacent_edgeweights = agent[edgeids][adjacent_edge_ids[dim]]

    dH = -(belief_options - old_belief) * np.sum(
        adjacent_edgeweights * adjacent_beliefs
    )

    p = 1.0 / (1.0 + np.exp(beta * dH))

    return p / p.sum()

def update_edge_weights(agent, eps):
    """Update all edge weights for an agent"""
    edgeweights = agent[edgeids]
    del_beliefs = agent[delbeliefids].reshape((tau, M)).mean(axis=0)
    return (
        edgeweights
        + eps * del_beliefs[edge_arr[:, 0]] * del_beliefs[edge_arr[:, 1]]
    )

def update_belief(agent, beta):
    """Update all belief values for an agent."""
    agent_prior_beliefs = np.copy(agent[beliefids])

    np.random.shuffle(beliefupdate_order)

    for dim in beliefupdate_order:
        probabilities = glauber_fast(dim, agent, beta)

        agent[beliefids[dim]] = np.random.choice(
            belief_options, p=probabilities
        )

    # Shift belief changes by one time step.
    agent[delbeliefids[M:]] = agent[delbeliefids[:-M]]

    # Store the latest belief changes.
    agent[delbeliefids[:M]] = agent[beliefids] - agent_prior_beliefs

    return agent
def get_metrics(agents):
    # Mean absolute edge strength across all connections
    bn_abs_meanedge_tot = np.abs(agents[:, edgeids]).mean(axis=1)

    # Mean absolute edge strength for connections to the focal belief
    bn_abs_meanedge_foc = np.abs(
        agents[:, edgeids][:, adjacent_edge_ids[focal]]
    ).mean(axis=1)

    # Expected influence on the focal belief from connected beliefs
    A = np.zeros((n_agents, M, M))

    for i, j in edge_list:
        A[:, i, j] = agents[:, edgeids][:, edge_lookup[(i, j)]]
        A[:, j, i] = A[:, i, j]

    expected_influence = (
        A[:, focal, :] * agents[:, beliefids]
    ).sum(axis=1)

    return (
        bn_abs_meanedge_tot,
        bn_abs_meanedge_foc,
        expected_influence,
    )
    
def fill_metrics(t, agents, params):
    absOm_tot, absOm_foc, expI = get_metrics(agents)

    # Mean absolute edge strength across the network
    agents[:, 2] = absOm_tot

    # Mean absolute edge strength connected to the focal belief
    agents[:, 3] = absOm_foc

    # Expected influence on the focal belief
    agents[:, 4] = expI

    # The focal belief itself
    agents[:, 5] = agents[:, beliefids[focal]]

    # Mean absolute value of all non-focal beliefs
    nonfocal_ids = [
        k for k in beliefids if k != beliefids[focal]
    ]

    agents[:, 6] = np.mean(
        np.abs(agents[:, nonfocal_ids]), axis=1
    )

    return agents

def get_output(snapshots):
    df_full = pd.DataFrame(snapshots, columns=columns)
    return df_full

def run_simulation(params):
    np.random.seed(params["seed"])

    eps = params["eps"]
    beta = params["beta"]

    agents = initialise_agents(params["init_w"])
    agents = fill_metrics(0, agents, params)

    snapshots = np.copy(agents)
    time_steps = np.arange(0, T + 1, 1)

    for t in time_steps[1:]:
        np.random.shuffle(agentids)

        for n in agentids:
            agents[n] = update_belief(agents[n], beta)
            agents[n, edgeids] = update_edge_weights(agents[n], eps)

        agents[:, 0] = t

        if t in track_times or t == time_steps[-1]:
            agents = fill_metrics(t, agents, params)
            snapshots = np.concatenate([snapshots, agents])

    output_df = get_output(snapshots)
    return output_df

def run_one(seed, init_w, beta, eps):
    results_folder = "simOut/"
    detailed_folder = results_folder + "detailed/"

    os.makedirs(detailed_folder, exist_ok=True)

    params = {
        "init_w": init_w,
        "beta": beta,
        "eps": eps,
        "seed": seed,
    }

    fname = (
        detailed_folder
        + f"sim_init_w{init_w:.2f}"
        + f"_beta{beta:.2f}"
        + f"_eps{eps:.2f}"
        + f"_seed{seed}_detailed.csv"
    )

    if seed % 25 == 0:
        print(fname + "...")

    # Do not rerun a simulation if its output already exists.
    if not os.path.isfile(fname):
        out = run_simulation(params)

        for k, v in params.items():
            out[k] = v

        out.to_csv(fname, index=False)
        print("Saved:", fname)

    return fname
# %%
# Main execution
if __name__ == "__main__":
    beta = 3.0
    init_w = 0.2

    eps_values = [0.0, 1.0]
    seeds = list(range(5))

    if detail and len(seeds) > 10:
        print("...this will take very long")
        quit()

    param_combis_withSeed = [
        (seed, init_w, beta, eps)
        for eps in eps_values
        for seed in seeds
    ]

    print("Running on jobs:", max(1, multiprocessing.cpu_count() - 1))

    Parallel(n_jobs=max(1, multiprocessing.cpu_count() - 1))(
        delayed(run_one)(seed, init_w, beta, eps)
        for seed, init_w, beta, eps in param_combis_withSeed
    )

    print("\nAll simulations completed.")