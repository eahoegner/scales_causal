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

import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from matplotlib import pyplot as plt
from scales_cv.analysis import define_mask, run_pcmci_analysis
from scales_cv.emulation import (
    causal_model,
    compare_dicts,
    rescale_links_coeffs,
    val_matrix_to_links_dict,
)
from scales_cv.util import (
    ar6_nodes_for_plotting,
    deseasonalise_dataframe,
    detrend_dataframe,
    prepare_input,
    select_data,
)
from tigramite.independence_tests.parcorr import ParCorr

# from tigramite.independence_tests.robust_parcorr import RobustParCorr
# from tigramite.independence_tests.gpdc import GPDC
# from tigramite.independence_tests.cmiknn import CMIknn

# %% [markdown]
# ## import and prepare data

# %%
input_folder = Path().resolve().parent / "outputs"
output_folder = "/Users/hoegner/Projects/scales/causal_analysis_flat10/emulations_v01/ssp370_cheating"
os.makedirs(Path(output_folder), exist_ok=True)
in_file = "tas_regional_monthly.csv"
global_file = "tas_full_monthly.csv"
experiment_dd = "esm-flat10"

# %%
data_wide, region_names = prepare_input(
    input_folder, in_file=in_file, sourceDD="ACCESS-ESM1-5", experiment_dd=experiment_dd
)

# %%
output_folder

# %%
names = region_names  # or a selection such as
# selection = ["time", "EPO", "SPO", "NWS", "WAN", "SSA", "SAO", "EAN", "NAO", "SIO", "SAU", "TIB", "WSB", "GIC"]
# names = ["EPO", "SPO", "NWS", "WAN", "SSA", "SAO", "EAN", "NAO", "SIO", "SAU", "TIB", "WSB", "GIC"]

# %%
# data_wide = data_wide[selection]
data_wide

# %%
plt.figure(figsize=(10, 3))
plt.plot(df_detrended["GIC"])
plt.plot(df_deseasonalised["GIC"])

# %%
df_detrended, df_trend = detrend_dataframe(data_wide, time_col="time", tau=48)


# %%
def deseasonalise_dataframe(df, time_col="time", period=12):
    """
    Remove a fixed mean seasonal cycle.
    Assumes time_col is integer-like so: season = time % period.
    """
    variable_cols = [c for c in df.columns if c != time_col]

    # seasonal index
    season_index = df[time_col].astype(int) % period

    # compute climatology
    climatology = df.groupby(season_index)[variable_cols].mean()

    # expand climatology to full series
    seasonal = climatology.iloc[season_index].reset_index(drop=True)

    # subtract
    deseasonalised = df.copy()
    deseasonalised[variable_cols] = df[variable_cols].values - seasonal.values

    # attach time column to seasonal
    seasonal.insert(0, time_col, df[time_col].values)

    return deseasonalised, seasonal


# %%
# if monthly, then
df_deseasonalised, df_seasonal = deseasonalise_dataframe(df_detrended)
data, std_devs = select_data(df_deseasonalised, names)

# if annual, then
# data, std_devs = select_data(df_detrended, names)

# %%
node_pos = ar6_nodes_for_plotting(names)

# %% [markdown]
# ## Causal analysis

# %%
tau_max = 5
alpha_level = 0.05
mask_sel = "MAM"

mask = define_mask(data, selection=mask_sel)

results, graph, Links, fig = run_pcmci_analysis(
    data=data,
    region_names=names,
    tau_max=tau_max,
    alpha_level=alpha_level,
    ci_test=ParCorr,
    mask_type=None,
    # mask=mask,
    tag="training",
    plot_timeseries=True,
    # output_folder=output_folder,
    # in_file=input_folder / in_file,
    sourceDD="ACCESS-ESM1-5",
    experiment_dd=f"{mask_sel}",
    node_pos=node_pos,
    verbosity=0,
)

# %%
Links[Links > 1] = 1

# %% [markdown]
# ## Emulation

# %%
emulated, em_df, em_std_devs, coeffs = causal_model(
    Links=Links, tau_max=tau_max, std_devs=std_devs, names=names, T=800, burn=300
)
# ,seed=411)

# %%
em_results, em_graph, em_Links, em_fig = run_pcmci_analysis(
    data=emulated,
    region_names=names,
    tau_max=tau_max,
    alpha_level=alpha_level,
    ci_test=ParCorr,
    tag="emulation",
    plot_timeseries=True,
    #  output_folder=output_folder,
    #  in_file=input_folder / in_file,
    sourceDD="ACCESS-ESM1-5",
    experiment_dd="esm-flat10",
    node_pos=node_pos,
    verbosity=0,
)

# %%
em_coeffs = val_matrix_to_links_dict(em_Links, tau_max=tau_max)
re_em_coeffs = rescale_links_coeffs(em_coeffs, np.array(em_std_devs))

# %%
plt.figure(figsize=(9, 2))
plt.scatter(names, std_devs / em_std_devs, marker="x", s=12, c="b")
plt.title("Standard deviations, original/emulated")
plt.ylim(0.5, 1.5)
plt.axhline(y=1, color="r", linestyle="--", alpha=0.8)
plt.xticks(rotation=60, fontsize=7)
plt.grid(alpha=0.2)

# %% [markdown]
# ## ensemble generation

# %%
ensemble_differences = []
emulations = []
all_links = []
std_dev_df = []

training_row = {"run_id": "original"}
for region, val in zip(names, std_devs):
    training_row[region] = val
std_dev_df.append(training_row)


for i in range(100):
    emulated, em_df, em_std_devs, coeffs = causal_model(
        Links=Links, tau_max=tau_max, std_devs=std_devs, names=names, T=300, burn=200
    )

    em_results, em_graph, em_Links, em_fig = run_pcmci_analysis(
        data=emulated,
        region_names=names,
        tau_max=tau_max,
        alpha_level=alpha_level,
        ci_test=ParCorr,
        tag=f"emulation_{i}",
        plot_timeseries=False,
        output_folder=output_folder,
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

    print(f"Emulating time series with run_id={i}")

    em_df["run_id"] = i
    em_df["time"] = em_df.index
    cols = ["run_id", "time"] + [
        c for c in em_df.columns if c not in ("run_id", "time")
    ]
    em_df = em_df[cols]
    emulations.append(em_df)

    row = {"run_id": i}
    for region, val in zip(names, em_std_devs):
        row[region] = val
    std_dev_df.append(row)

df_diff = pd.concat(ensemble_differences, ignore_index=True)
df_diff.to_csv(Path(output_folder) / "ensemble_link_differences.csv", index=False)

df_all = pd.concat(all_links, ignore_index=True)
df_all.to_csv("ensemble_all_links.csv", index=False)

emulations_df = pd.concat(emulations, ignore_index=True)
emulations_df.to_csv(Path(output_folder) / "ensemble_emulations.csv")

std_df = pd.DataFrame(std_dev_df)
std_df.to_csv(Path(output_folder) / "ensemble_std_devs.csv", index=False)

# %% [markdown]
# ## build seasonal model

# %%
tau_max = 10
alpha_level = 0.01

season_links_dict = {}

for mask_sel in ["DJF", "MAM", "JJA", "SON"]:
    mask = define_mask(data, selection=mask_sel)

    results, graph, Links_season, fig = run_pcmci_analysis(
        data=data,
        region_names=names,
        tau_max=tau_max,
        alpha_level=alpha_level,
        ci_test=ParCorr,
        mask_type="y",
        mask=mask,
        tag=f"training_{mask_sel}",
        plot_timeseries=False,
        sourceDD="ACCESS-ESM1-5",
        experiment_dd=f"{mask_sel}",
        node_pos=node_pos,
        verbosity=0,
    )

    # Convert PCMCI val_matrix to dict format
    season_links_dict[mask_sel] = val_matrix_to_links_dict(
        Links_season, tau_max=tau_max
    )

# %%
seasonal_A = build_seasonal_A_matrices(
    season_links_dict, n_vars=len(names), tau_max=tau_max
)

# %%
len(ssp370) / 12


# %%
def month_to_season(month):
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


years = 86
months_total = years * 12
season_seq = []
start_month = 1
for m in range(months_total + 200):  # + burn to ensure length >= T+burn
    month = ((start_month - 1 + m) % 12) + 1
    season_seq.append(month_to_season(month))


# %%
def make_monthly_season_seq(start_month, months_total, burn=200):
    seq = []
    for m in range(months_total + burn):
        month = ((start_month - 1 + m) % 12) + 1
        seq.append(month_to_season(month))
    return seq


# %%
Y_sim = simulate_seasonal_VAR(
    seasonal_A=seasonal_A,
    std_devs=std_devs,
    season_seq=season_seq,
    T=months_total,
    burn=200,
    verbose=True,
)

df_sim = pd.DataFrame(Y_sim, columns=names)

# %%
df_sim

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp_deseasonalised["EPO"])
plt.plot(df_sim["EPO"])

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp_deseasonalised["NAO"])
plt.plot(df_sim["NAO"])

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp_deseasonalised["NPO"])
plt.plot(df_sim["NPO"])

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp_deseasonalised["ARO"])
plt.plot(df_sim["ARO"])

# %% [markdown]
# ## cheating "emulation"

# %%
ssp370 = xr.open_dataset(
    "/Users/hoegner/Projects/data/CMIP6/tas_Amon_ACCESS-ESM1-5_ssp370_r31i1p1f1_region_aggregates.nc"
)

# %%
ssp370_alt = xr.open_dataset(
    "/Users/hoegner/Projects/data/CMIP6/tas_Amon_ACCESS-ESM1-5_ssp370_r30i1p1f1_region_aggregates.nc"
)

# %%
ssp370 = ssp370["tas"].to_pandas().reset_index()

# %%
ssp370_alt = ssp370_alt["tas"].to_pandas().reset_index()

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp370["GIC"])

# %%
time_axis = ssp370["time"]

# %%
ssp370_data = ssp370.drop(columns=["time"])
ssp370_data["time"] = ssp370.index

# %%
ssp370_data

# %%
ssp_detrended, ssp_trend = detrend_dataframe(ssp370_data, time_col="time", tau=48)

# %%
ssp_deseasonalised, ssp_seasonal = deseasonalise_dataframe(ssp_detrended)

# %%
ssp_detrended

# %%
plt.plot(ssp_trend["CAR"])

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp370["ARO"])
# plt.plot(ssp_trend["ARO"] + df_seasonal["ARO"][0:len(ssp_trend)] + df_sim["ARO"]);

# %%
plt.figure(figsize=(10, 3))
# plt.plot(ssp370["EPO"])
plt.plot(ssp_trend["EPO"] + df_seasonal["EPO"][0 : len(ssp_trend)] + df_sim["EPO"])
plt.plot(ssp370_alt["EPO"])

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp370["NPO"])
plt.plot(ssp_trend["NPO"] + df_seasonal["NPO"][0 : len(ssp_trend)] + df_sim["NPO"])

# %%
plt.figure(figsize=(10, 3))
plt.plot(ssp370["GIC"])
plt.plot(ssp_trend["GIC"] + df_seasonal["GIC"][0 : len(ssp_trend)] + df_sim["GIC"])

# %%
output_folder = Path(output_folder)

# %%
full_emulation = ssp_trend + df_seasonal[0 : len(ssp_trend)] + df_sim

# %%
full_emulation["time"] = time_axis
full_emulation = full_emulation[
    ["time"] + [c for c in full_emulation.columns if c != "time"]
]

# %%
value_cols = [c for c in full_emulation.columns if c != "time"]

n_rows, n_cols = 29, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 40), sharex=True)
axes = axes.flatten()

for ax, col in zip(axes, value_cols):
    ax.plot(full_emulation["time"], full_emulation[col])
    ax.plot(ssp370["time"], ssp370[col], alpha=0.8)
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=6)

# Hide unused axes (in case fewer than 58 columns)
for ax in axes[len(value_cols) :]:
    ax.set_visible(False)

fig.tight_layout()
plt.savefig(output_folder / "compare_ssp370_emulation_taumax10.png")
plt.show()

# %%
df = full_emulation - ssp370
df["time"] = time_axis

# %%
ssp_diffs = ssp370_alt - ssp370

# %%
value_cols = [c for c in df.columns if c != "time"]

n_rows, n_cols = 29, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 40), sharex=True)
axes = axes.flatten()

for ax, col in zip(axes, value_cols):
    ax.plot(df["time"], df[col])
    ax.plot(df["time"], ssp_diffs[col], alpha=0.8)
    #    ax.plot(df["time"], df_deseasonalised[col][0:len(df)])
    ax.set_title(col, fontsize=8)
    ax.tick_params(labelsize=6)

# Hide unused axes (in case fewer than 58 columns)
for ax in axes[len(value_cols) :]:
    ax.set_visible(False)

fig.tight_layout()
plt.savefig(output_folder / "difference_ssp370_emulation_taumax10.png")
plt.show()

# %%
ssp370

# %%
ssp_deseasonalised

# %%
ssp_std_devs = np.std(ssp_deseasonalised.drop(columns=["time"]))

# %%
plt.figure(figsize=(9, 2))
plt.scatter(names, ssp_std_devs / em_std_devs, marker="x", s=12, c="b")
plt.title("Standard deviations, original/emulated")
# plt.ylim(0.5,1.5)
plt.axhline(y=1, color="r", linestyle="--", alpha=0.8)
plt.xticks(rotation=60, fontsize=7)
plt.grid(alpha=0.2)

# %%
error = np.array(np.std(df.drop(columns=["time"])))

# %%
full_emulation.to_csv(
    output_folder / "ACCESS-ESM1-5_cheat_emulation_ssp370_r31i1p1f1_varflat10.csv"
)

# %% [markdown]
# ## add annual VAR?

# %%
df_regions = df_detrended.drop(columns="time")

# Compute annual mean by taking the mean of every 12 rows
annual = df_regions.groupby(df_regions.index // 12).mean()

# %%
names = regionmask.defined_regions.ar6.ocean.abbrevs

# %%
annual_df = annual[names]
annual_df.insert(0, "time", range(len(annual_df)))

# %%
annual_df

# %%
data, ocean_std_devs = select_data(annual_df, names)

# %%
node_pos = ar6_nodes_for_plotting(names)

# %%
tau_max = 5
alpha_level = 0.05

ocean_results, ocean_graph, ocean_Links, ocean_fig = run_pcmci_analysis(
    data=data,
    region_names=names,
    tau_max=tau_max,
    alpha_level=alpha_level,
    ci_test=ParCorr,
    tag="ocean",
    plot_timeseries=True,
    #  output_folder=output_folder,
    #  in_file=input_folder / in_file,
    sourceDD="ACCESS-ESM1-5",
    experiment_dd="esm-flat10",
    node_pos=node_pos,
    verbosity=0,
)

# %%
ocean_emulated, ocean_em_df, ocean_em_std_devs, ocean_coeffs = causal_model(
    Links=ocean_Links,
    tau_max=tau_max,
    std_devs=ocean_std_devs,
    names=names,
    T=300,
    burn=500,
)
# ,seed=411)

# %%
tau_max = 5
alpha_level = 0.05

ocean_em_results, ocean_em_graph, ocean_em_Links, ocean_em_fig = run_pcmci_analysis(
    data=ocean_emulated,
    region_names=names,
    tau_max=tau_max,
    alpha_level=alpha_level,
    ci_test=ParCorr,
    tag="ocea_emulated",
    plot_timeseries=False,
    #  output_folder=output_folder,
    #  in_file=input_folder / in_file,
    sourceDD="ACCESS-ESM1-5",
    experiment_dd="esm-flat10",
    node_pos=node_pos,
    verbosity=0,
)

# %%
ocean_em_df

# %%
monthly_ocean_em_df = pd.DataFrame(
    np.repeat(ocean_em_df[-86:].values, 12, axis=0), columns=ocean_em_df[-86:].columns
)
monthly_ocean_em_df

# %% [markdown]
# ## COMBINE ALL

# %%
seasonality = df_seasonal[0 : len(ssp_trend)]

# %%
seasonal_with_annual = seasonality.add(monthly_ocean_em_df, fill_value=0)

# %%
plt.plot(seasonal_with_annual["EPO"])

# %%
full_emulation_with_oceans = ssp_trend + seasonal_with_annual + df_sim

# %%
plt.figure(figsize=(12, 3))
plt.plot(full_emulation_with_oceans["NAO"], c="b", label="with oceans")
plt.plot(full_emulation["NAO"], c="r", label="emulation")
plt.plot(ssp370_alt["NAO"], c="c", label="original")
plt.legend()

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)  # 3 panels stacked

# First panel
axes[0].plot(
    full_emulation_with_oceans["EPO"],
    c="b",
    label="emulation with annual variability for oceans",
)
axes[0].set_title("EPO")
axes[0].legend()

# Second panel
axes[1].plot(full_emulation["EPO"], c="r", label="emulation, monthly only")
axes[1].legend()

# Third panel
axes[2].plot(ssp370_alt["EPO"], c="c", label="original")
axes[2].set_xlabel("Time")
axes[2].legend()

# %%
np.std(ssp370_alt["EPO"])

# %%
np.std(full_emulation_with_oceans["EPO"])

# %%
np.std(full_emulation["EPO"])
