# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline
import os
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import regionmask
from matplotlib import pyplot as plt
from scales_cv.analysis import run_pcmci_analysis
from scales_cv.emulation import (
    aggregate_links_weighted,
    causal_model,
    compare_dicts,
    rescale_links_coeffs,
    val_matrix_to_links_dict,
)
from scales_cv.util import (
    create_nodes_for_plotting,
    detrend_dataframe,
    deseasonalise_dataframe,
    select_data,
    split_overshoot,
)
from tigramite.independence_tests.parcorr import ParCorr

import pickle

# from tigramite.independence_tests.robust_parcorr import RobustParCorr
# from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.cmiknn import CMIknn

# %% [markdown]
# # Import and prepare data

# %% [markdown]
# ## import

# %%
model = "ssp245_ensemble40_monthly_2500m_m0_Jan.pkl"
#model = "ssp245_ensemble40_monthly_2499m_m0_Jan_v0.2.pkl"
scales_path = f"/Users/hoegner/Projects/scales/emulator/{model}"
y_pred_ensemble = pickle.load( open(scales_path, "rb") ) #dim = n_samples,n_months,n_regions

#if freq == "monthly":
#    input_df["time"] = input_df["time"].apply(
#        lambda x: cftime.DatetimeGregorian(int(x[:4]), int(x[5:7]), 1)
#    )

# %%
len(y_pred_ensemble[1][1])

# %%
experiment_dd = "ssp245_generated"

output_folder = Path("/Users/hoegner/Projects/scales/emulator/analysis")

# %%
# Get AR6 abbreviations in correct mask order
ar6_regions = regionmask.defined_regions.ar6.all
ar6_abbrevs = ar6_regions.abbrevs
ar6_numbers = ar6_regions.numbers

names = ar6_abbrevs  # or a selection such as ["EPO", "NPO", "NAO", "SAM", "NWN"]
#names = regionmask.defined_regions.ar6.ocean.abbrevs

# %%
n_members = len(y_pred_ensemble)
n_time = (y_pred_ensemble[0].shape[0] // 12) *12
n_regions = y_pred_ensemble[0].shape[1]

members = [f"member_{i}" for i in range(n_members)]
times = np.arange(n_time)  # or your actual time array
regions = sorted(ar6_regions.abbrevs)

# %%
df_long = pd.DataFrame(
    {
        "member": np.repeat(members, n_time * n_regions),
        "time": np.tile(np.repeat(times, n_regions), n_members),
        "region": np.tile(regions, n_members * n_time),
        "value": np.array(y_pred_ensemble[:, :n_time, :]).reshape(-1),
    }
)

# %%
df_wide = (
    df_long
    .pivot(index=["member", "time"], columns="region", values="value")
    .reset_index()
)

# %%
df_wide[df_wide["member"]=="member_1"]

# %%
fig = plt.figure(figsize=(12,4))
for k in range(0,39):
    plt.plot(df_wide[df_wide["member"]==f"member_{k}"]["time"], 
             df_wide[df_wide["member"]==f"member_{k}"]["NAO"], alpha=.1);

# %%
df_annual = (
    df_wide
    .assign(year=lambda d: d["time"] // 12)
    .groupby(["member", "year"], as_index=False)[regions]
    .mean()
)

df_annual

# %% [markdown]
# ## detrend

# %%
detrended = {}
time_col = "year"

for member, df_member in df_annual.groupby("member"):
    # Other columns to keep (metadata, etc.)
    other_cols = [c for c in df_member.columns if c not in names + [time_col]]

    # Only detrend the columns in `names`
    df_detrended, _ = detrend_dataframe(
        df_member[[time_col] + names], time_col=time_col, tau=20, verbose=True
    )

    # Add back the other columns unchanged
    for c in other_cols:
        df_detrended[c] = df_member[c].values

    # Preserve original column order
    df_detrended = df_detrended[df_member.columns]

    # Store in dictionary
    detrended[member] = df_detrended

# %% [markdown]
# ## define regions for plotting

# %%
node_pos = create_nodes_for_plotting(names)

# %% [markdown]
# # Causal analysis

# %%
keys = list(detrended.keys())

# %%
plt.plot(detrended[keys[1]]["year"], detrended[keys[1]]["ARO"])

# %%
#input_folder = Path("/Users/hoegner/Projects/scales/emulator")
#in_file = "ssp245_ensemble40_monthly_2500m.pkl"
filename = Path("ssp245_scales")

tau_max = 5
alpha_level = 0.05
Links_dict = {}


for i in keys:
    temp = detrended[i]
    data, std_devs = select_data(temp, names)

    results, graph, Links, fig = run_pcmci_analysis(
        data=data,
        region_names=names,
        tau_max=tau_max,
        alpha_level=alpha_level,
        ci_test=ParCorr,
        # mask_type='y',
        # mask=mask,
        tag="ssp245_generated",
        plot_timeseries=True,
        output_folder=output_folder,
        in_file=filename,
        sourceDD=f"{i}",
        experiment_dd=experiment_dd,
        node_pos=node_pos,
        verbosity=0,
    )

    Links_dict[i] = Links

# %%
agg, frequency = aggregate_links_weighted(
    Links_dict, min_freq_to_keep=6, method="median", weight_mode="freq/num", gamma=1
)

# %% [markdown]
# # Emulate

# %% [markdown]
# ## Generative Causal Model

# %%
emulated, emulated_df, em_std_devs, coeffs = causal_model(
    Links=agg,
    tau_max=tau_max,
    std_devs=std_devs,
    names=names,
    T=500,
    burn=200,
    seed=112
)

# %%
emulated
# np.save("emulation.npy", emulated)

# %%
alpha_level = 0.07
filename = Path("ssp245_scales")

em_results, em_graph, em_Links, em_fig = run_pcmci_analysis(
    data=emulated,
    region_names=names,
    tau_max=tau_max,
    alpha_level=alpha_level,
    ci_test=ParCorr,
    tag="emulation",
    plot_timeseries=True,
    output_folder=output_folder,
    in_file=filename,
    sourceDD="ACCESS-ESM1-5",
    experiment_dd=experiment_dd,
    node_pos=node_pos,
    verbosity=0,
)

# %%
em_coeffs = val_matrix_to_links_dict(em_Links, tau_max=tau_max)
re_em_coeffs = rescale_links_coeffs(em_coeffs, np.array(em_std_devs))

# %%
compare_dicts(coeffs, re_em_coeffs, names, 1, tolerance=0.1)

# %%
plt.figure(figsize=(9, 2))
plt.scatter(names, std_devs / em_std_devs, marker="x", s=12, c="b")
plt.title("Standard deviations, original/emulated")
plt.ylim(0.5, 1.5)
plt.axhline(y=1, color="r", linestyle="--", alpha=0.8)
plt.xticks(rotation=60, fontsize=7)
plt.grid(alpha=0.2)

# %% [markdown]
# ## ensemble

# %%
ensemble_differences = []
all_links = []
std_dict = {"training": std_devs}

ensemble_folder = output_folder / f"ensemble_{experiment_dd}"
os.makedirs(Path(ensemble_folder), exist_ok=True)

for i in range(30):
    emulated, emulated_df, em_std_devs, coeffs = causal_model(
        Links=Links, tau_max=tau_max, std_devs=std_devs, names=names, T=500, burn=200
    )

    em_results, em_graph, em_Links, em_fig = run_pcmci_analysis(
        data=emulated,
        region_names=names,
        tau_max=tau_max,
        alpha_level=alpha_level,
        ci_test=ParCorr,
        tag=f"emulation_{i}",
        plot_timeseries=False,
        output_folder=ensemble_folder,
        in_file=input_folder / in_file,
        sourceDD="ACCESS-ESM1-5",
        experiment_dd=experiment_dd,
        node_pos=node_pos,
        verbosity=0,
        show_figure=False,
    )

    em_coeffs = val_matrix_to_links_dict(em_Links, tau_max=tau_max)
    re_em_coeffs = rescale_links_coeffs(em_coeffs, np.array(em_std_devs))

    diff, df_all_links = compare_dicts(
        coeffs, re_em_coeffs, names, run_id=i, tolerance=0.1
    )
    ensemble_differences.append(diff)
    all_links.append(df_all_links)

    std_dict[f"run_{i}"] = em_std_devs
    print(f"Emulating time series with run_id={i}")


df_diff = pd.concat(ensemble_differences, ignore_index=True)
df_diff.to_csv(Path(ensemble_folder) / "ensemble_link_differences.csv", index=False)

df_all = pd.concat(all_links, ignore_index=True)
df_all.to_csv(Path(ensemble_folder) / "ensemble_all_links.csv", index=False)


regions = std_dict["training"].index
# convert all std_dev dictionary entries to Series
std_dict_fixed = {
    k: (v if isinstance(v, pd.Series) else pd.Series(v, index=regions))
    for k, v in std_dict.items()
}
std_df = pd.concat(std_dict_fixed, axis=1)
std_df.index.name = "region"
std_df.to_csv(Path(ensemble_folder) / "ensemble_std_devs.csv")

# %% [markdown]
# ## monthly

# %%
out_deseasonalised = []
out_seasonal = []

for m, g in df_wide.groupby("member"):
    g = g.copy()

    # drop non-numeric columns except time_col
    g = g.drop(columns=["member"])

    ds, s = deseasonalise_dataframe(
        g,
        time_col="time",
        period=12
    )

    # reattach metadata
    ds["member"] = m
    s["member"] = m

    out_deseasonalised.append(ds)
    out_seasonal.append(s)

deseasonalised = pd.concat(out_deseasonalised, ignore_index=True)
seasonal = pd.concat(out_seasonal, ignore_index=True)

# %%
from scipy.ndimage import gaussian_filter1d

def detrend_gaussian(df, time_col="time", tau=40):
    var_cols = [c for c in df.columns if c != time_col]
    sigma = tau  # tau already in time units (months)

    df_trend = df.copy()
    for c in var_cols:
        df_trend[c] = gaussian_filter1d(df[c].values, sigma=sigma, mode="nearest")

    df_detrended = df.copy()
    df_detrended[var_cols] = df[var_cols] - df_trend[var_cols]

    return df_detrended, df_trend


# %%
detrended_monthly = {}
time_col = "time"

for member, df_member in deseasonalised.groupby("member"):
    # Other columns to keep (metadata, etc.)
    other_cols = [c for c in df_member.columns if c not in names + [time_col]]

    # Only detrend the columns in `names`
    df_detrended, _ = detrend_gaussian(
        df_member[[time_col] + names], time_col=time_col, tau=40
    )

    # Add back the other columns unchanged
    for c in other_cols:
        df_detrended[c] = df_member[c].values

    # Preserve original column order
    df_detrended = df_detrended[df_member.columns]

    # Store in dictionary
    detrended_monthly[member] = df_detrended

# %%
plt.plot(detrended_monthly["member_0"]["time"], detrended_monthly["member_0"]["NWN"])

# %%
# Your data shape
n_rows, n_cols = 2496, 58

# Pattern to repeat along rows (length 12)
#pattern = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  # DJF
#pattern = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1] # MAM
#pattern = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1] # JJA
pattern = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1] # SON

# Repeat pattern enough times to cover all rows
repeats = int(np.ceil(n_rows / len(pattern)))
full_pattern = np.tile(pattern, repeats)[:n_rows]  # trim to exact row count

# Broadcast to all columns
mask = np.tile(full_pattern[:, None], (1, n_cols))

print(mask.shape)  # (1980, 58)
print(mask)

# %%
keys = list(detrended_monthly.keys())

# %%
output_folder= Path('/Users/hoegner/Projects/scales/emulator/analysis/monthly')

# %%
detrended_monthly[keys[1]]

# %%
#input_folder = Path("/Users/hoegner/Projects/scales/emulator")
#in_file = "ssp245_ensemble40_monthly_2500m.pkl"
filename = Path("SON_ssp245_scales")

tau_max = 5
alpha_level = 0.05
Links_dict = {}


for i in keys:
    temp = detrended_monthly[i]
    data, std_devs = select_data(temp, names)

    results, graph, Links, fig = run_pcmci_analysis(
        data=data,
        region_names=names,
        tau_max=tau_max,
        alpha_level=alpha_level,
        ci_test=ParCorr,
        mask_type='y',
        mask=mask,
        tag="ssp245_generated",
        plot_timeseries=False,
        output_folder=output_folder,
        in_file=filename,
        sourceDD=f"{i}",
        experiment_dd=experiment_dd,
        node_pos=node_pos,
        verbosity=0,
    )

    Links_dict[i] = Links

# %%
agg, frequency = aggregate_links_weighted(
    Links_dict, min_freq_to_keep=6, method="median", weight_mode="freq/num", gamma=1
)

# %%
emulated, emulated_df, em_std_devs, coeffs = causal_model(
    Links=agg,
    tau_max=tau_max,
    std_devs=std_devs,
    names=names,
    T=500,
    burn=200,
    seed=112
)

# %%
alpha_level = 0.07
filename = Path("ssp245_scales")

em_results, em_graph, em_Links, em_fig = run_pcmci_analysis(
    data=emulated,
    region_names=names,
    tau_max=tau_max,
    alpha_level=alpha_level,
    ci_test=ParCorr,
    tag="SON_scales_emulation",
    plot_timeseries=True,
    output_folder=output_folder,
    in_file=filename,
    sourceDD="ACCESS-ESM1-5",
    experiment_dd=experiment_dd,
    node_pos=node_pos,
    verbosity=0,
)
