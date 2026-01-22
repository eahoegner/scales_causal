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
from pathlib import Path

import cftime
import numpy as np
import pandas as pd

# %matplotlib inline
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt

# %%
folder = Path(
    "/Users/hoegner/Projects/data/ESGF/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp534-over-full/aggregated"
)

# %%
pd.read_csv(
    folder / "tas_Amon_ACCESS-ESM1-5_ssp534-over_r1i1p1f2_gn_201501-230012.csv",
    index_col=0,
)

# %%
dfs = []

for file in folder.glob("*.csv"):
    member = file.stem.split("_")[-3]

    df = pd.read_csv(file, index_col=0, dtype={"time": "string"})

    # example assumes ISO-like dates: YYYY-MM or YYYY-MM-DD
    df["time"] = df["time"].apply(
        lambda x: cftime.DatetimeGregorian(int(x[:4]), int(x[5:7]), 1)
    )

    df["member"] = member
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# %%
df

# %%
np.unique(df["member"])

# %%
df = df.sort_values(["member", "time"])
df_monthly = df.reset_index(drop=True)


df["year"] = df["time"].apply(lambda t: t.year + (t.month - 1) / 12)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df, x="year", y="GLOBAL", hue="member", estimator=None, alpha=0.1, legend=False
)

plt.xlabel("Time")
plt.ylabel("GLOBAL")
plt.title("GLOBAL over time by ensemble member")
plt.tight_layout()
plt.show()

# %%
df_short = df[-500:]
plt.figure(figsize=(10, 6))

df_short["year"] = df_short["time"].apply(lambda t: t.year + (t.month - 1) / 12)

sns.lineplot(
    data=df_short,
    x="year",
    y="ARO",
    hue="member",
    estimator=None,
    alpha=0.6,
    legend=False,
)

plt.xlabel("Time")
plt.ylabel("GLOBAL")
plt.title("GLOBAL over time by ensemble member")
plt.tight_layout()
plt.show()

# %%
df["year"] = df["time"].apply(lambda t: t.year)
df_numeric = df.drop(columns=["time"])

# convert all remaining value columns to numeric
value_cols = df_numeric.columns.difference(["member", "year"])
df_numeric[value_cols] = df_numeric[value_cols].apply(pd.to_numeric, errors="coerce")

# compute annual means
annual_df = df_numeric.groupby(["member", "year"], as_index=False).mean()
annual_df

# %%
annual_df = annual_df.sort_values(["member", "year"])

mean_df = df.groupby("year", as_index=False)["GLOBAL"].mean()

mean_df["GLOBAL_smooth"] = mean_df["GLOBAL"].rolling(window=5, center=True).mean()

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=annual_df,
    x="year",
    y="GLOBAL",
    hue="member",
    estimator=None,
    alpha=0.5,
    legend=False,
)

# Ensemble mean (on top)
sns.lineplot(
    data=mean_df,
    x="year",
    y="GLOBAL",
    color="black",
    linewidth=2,
    label="Ensemble mean",
)

# Ensemble mean (on top)
sns.lineplot(
    data=mean_df,
    x="year",
    y="GLOBAL_smooth",
    color="lightgrey",
    linewidth=1.5,
    alpha=0.8,
    label="Ensemble mean",
)

plt.xlabel("Time")
plt.ylabel("GLOBAL")
plt.title("GLOBAL over time by ensemble member")
plt.tight_layout()
plt.show()

# %%
# derive mean peak year
mean_df.loc[mean_df["GLOBAL_smooth"].idxmax(), "year"]

# %%
# columns to plot (all regions)
region_cols = [c for c in annual_df.columns if c not in ["year", "member", "GLOBAL"]]

# melt to long format
long_df = annual_df.melt(
    id_vars=["member", "year"],
    value_vars=region_cols,
    var_name="region",
    value_name="value",
)

mean_df = long_df.groupby(["region", "year"], as_index=False)["value"].mean()


# %%
def plot_ensemble_panels(
    annual_df,
    value_cols=None,
    ncols=2,
    member_alpha=0.3,
    mean_line_color="black",
    mean_line_width=2,
    show_quantiles=False,
):
    """
    Plot ensemble member time series faceted by region (or GLOBAL + regions).

    Parameters
    ----------
    - df: pandas DataFrame with columns 'member', 'year', plus value_cols
    - value_cols: list of columns to plot; if None, all except ['member', 'year']
    - ncols: number of columns in facet grid
    - member_alpha: transparency for ensemble member lines
    - mean_line_color: color of ensemble mean line
    - mean_line_width: width of ensemble mean line
    - show_quantiles: if True, show 5-95% shaded range
    """
    # Default value columns
    if value_cols is None:
        value_cols = [c for c in df.columns if c not in ["member", "year"]]

    # Melt to long format
    long_df = df.melt(
        id_vars=["member", "year"],
        value_vars=value_cols,
        var_name="region",
        value_name="value",
    )

    # Compute ensemble mean
    mean_df = long_df.groupby(["region", "year"], as_index=False)["value"].mean()

    # Optional: 5-95% quantiles
    if show_quantiles:
        quant_df = (
            long_df.groupby(["region", "year"])["value"]
            .quantile([0.05, 0.95])
            .unstack(level=2)
            .reset_index()
        )

    # FacetGrid
    g = sns.FacetGrid(
        long_df, col="region", col_wrap=ncols, height=3.5, aspect=2, sharey=False
    )

    # Plot ensemble member lines
    g.map_dataframe(
        sns.lineplot,
        x="year",
        y="value",
        hue="member",
        estimator=None,
        alpha=member_alpha,
        legend=False,
    )

    # Overlay mean line (ensure correct region-to-axis mapping)
    region_order = long_df["region"].unique()
    for ax, region in zip(g.axes.flatten(), region_order):
        # ensemble mean line
        mean_region = mean_df[mean_df["region"] == region].sort_values("year")
        ax.plot(
            mean_region["year"],
            mean_region["value"],
            color=mean_line_color,
            linewidth=mean_line_width,
        )

        # optional quantile shading
        if show_quantiles:
            q = quant_df[quant_df["region"] == region].sort_values("year")
            ax.fill_between(q["year"], q[0.05], q[0.95], color="gray", alpha=0.2)

    # Axis labels and titles
    g.set_axis_labels("Year", "")
    g.set_titles("{col_name}")
    plt.tight_layout()

    return g.fig


# %%
fig = plot_ensemble_panels(annual_df, value_cols=None, ncols=4, show_quantiles=True)

# %%
outpath = Path(
    "/Users/hoegner/Projects/data/ESGF/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp534-over-full/plots/"
)
outpath.mkdir(parents=True, exist_ok=True)

# %%
fig.savefig(outpath / "annual_regional_plots_HR.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# %%
def plot_members_one_region(df, region, ncols=4):
    plot_df = df[["member", "year", region]].copy()
    plot_df = plot_df.sort_values(["member", "year"])

    g = sns.FacetGrid(
        plot_df, col="member", col_wrap=ncols, height=2.5, aspect=1.6, sharey=True
    )

    g.map_dataframe(sns.lineplot, x="year", y=region, estimator=None)

    g.set_axis_labels("Year", region)
    g.set_titles("{col_name}")
    plt.tight_layout()

    return g.fig


# %%
regions = [c for c in df.columns if c not in ["year", "member"]]

for region in regions:
    fig = plot_members_one_region(df, region=region, ncols=4)

    fig.savefig(outpath / f"{region}_annual_members.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# %%
pictrl = xr.open_dataset(
    "/Users/hoegner/Projects/data/CMIP6/ACCESS-ESM1-5/pictrl/tas_Amon_ACCESS-ESM1-5_piControl_r1i1p1f1_region_aggregates.nc"
)

# %%
pictrl_df = pictrl["tas"].to_pandas().reset_index()

pictrl_df["year"] = pictrl_df["time"].apply(lambda t: t.year)
pictrl_numeric = pictrl_df.drop(columns=["time"])

# convert all remaining value columns to numeric
value_cols = pictrl_numeric.columns.difference(["year"])
pictrl_numeric[value_cols] = pictrl_numeric[value_cols].apply(
    pd.to_numeric, errors="coerce"
)

# compute annual means
annual_pictrl = pictrl_numeric.groupby(["year"], as_index=False).mean()
annual_pictrl

# %%
long_df = annual_pictrl.melt(
    id_vars="year", value_vars=value_cols, var_name="region", value_name="value"
)


# FacetGrid
g = sns.FacetGrid(long_df, col="region", col_wrap=4, height=3.5, aspect=2, sharey=False)

# Plot ensemble member lines
g.map_dataframe(sns.lineplot, x="year", y="value", estimator=None, legend=False)

# Overlay mean line (ensure correct region-to-axis mapping)
region_order = long_df["region"].unique()

# Axis labels and titles
g.set_axis_labels("Year", "")
g.set_titles("{col_name}")
plt.tight_layout()

g.savefig(outpath / "pictrl_annual.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## annual causal analysis in three parts

# %%
annual_df_up = annual_df[annual_df["year"] <= 2060]
annual_df_down = annual_df[(annual_df["year"] > 2060) & (annual_df["year"] < 2170)]
annual_df_stab = annual_df[annual_df["year"] >= 2170]

# %%
monthly_df_up = df_monthly[df_monthly["time"].apply(lambda t: t.year <= 2060)]
monthly_df_down = df_monthly[
    df_monthly["time"].apply(lambda t: (t.year > 2060) and (t.year <= 2170))
]
monthly_df_stab = df_monthly[df_monthly["time"].apply(lambda t: t.year >= 2170)]

# %%
monthly_df_stab

# %%
output_folder = Path().resolve().parent / "outputs" / "SSP5-34-EXT"
annual_df.to_csv(Path(output_folder / "SSP5-34-EXT_annual.csv"), index=False)
df_monthly.to_csv(Path(output_folder / "SSP5-34-EXT_monthly.csv"), index=False)
