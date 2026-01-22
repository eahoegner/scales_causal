import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import regionmask


def weights_calculate(x0, X, tau):
    """Gaussian kernel weights for local regression."""
    return np.exp(-np.sum((X - x0) ** 2, axis=1) / (2 * tau**2))


def local_weighted_regression(x0, X, Y, tau):
    """Locally weighted linear regression at point x0."""
    # ensure x0 is scalar
    x0 = float(np.squeeze(x0))

    X = np.c_[np.ones(len(X)), X]  # add intercept
    x0_vec = np.array([1, x0])
    W = np.diag(weights_calculate(np.array([[x0]]), X[:, 1:], tau))
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ Y)
    return float(x0_vec @ theta)


def detrend_dataframe(df, time_col="time", tau=20, verbose=True):
    """
    Detrend each column in df using local weighted regression (Gaussian kernel).
    """
    time_vals = df[time_col].to_numpy().reshape(-1, 1)
    variable_cols = [c for c in df.columns if c != time_col]

    df_trend = pd.DataFrame({time_col: df[time_col]})
    df_detrended = pd.DataFrame({time_col: df[time_col]})

    for col in variable_cols:
        Y = df[col].to_numpy()
        if verbose:
            print(f"Detrending {col} ...", end="\r")
        trend_vals = np.array(
            [local_weighted_regression(x0, time_vals, Y, tau) for x0 in time_vals]
        )
        df_trend[col] = trend_vals
        df_detrended[col] = Y - trend_vals

    if verbose:
        print(f"\n✅ Detrending complete. tau={tau}")
    return df_detrended, df_trend


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


def prepare_input(
    input_folder,
    in_file="tas_regional_yearly.csv",
    sourceDD="ACCESS-ESM1-5",
    experiment_dd="esm-flat10-zec",
):
    """
    Reads and prepares regional temperature data for a given model and experiment.

    Returns
    -------
    data_wide : pd.DataFrame
        DataFrame with numeric, sorted 'time' column and region columns.
    region_names : list
        List of region names.
    """
    in_file = input_folder / in_file
    variable = pd.read_csv(in_file)

    # Set hierarchical index
    variable = variable.set_index(
        ["variable", "region", "experiment_dd", "sourceDD", "unit"]
    )

    # Extract data for specified source and experiment
    example = variable.loc[
        (variable.index.get_level_values("sourceDD") == sourceDD)
        & (variable.index.get_level_values("experiment_dd") == experiment_dd)
    ].dropna(how="all", axis="columns")

    # Reshape to long format
    df_long = example.stack().reset_index()
    df_long = df_long.rename(
        columns={df_long.columns[-1]: "value", df_long.columns[-2]: "time"}
    )

    # Ensure time is numeric
    df_long["time"] = pd.to_numeric(df_long["time"], errors="coerce")

    # Drop rows where conversion failed
    df_long = df_long.dropna(subset=["time"])

    # Convert to integer if appropriate
    if (df_long["time"] % 1 == 0).all():
        df_long["time"] = df_long["time"].astype(int)

    # Pivot to wide format
    data_wide = df_long.pivot(index="time", columns="region", values="value")

    # Sort by numeric time
    data_wide = data_wide.sort_index().reset_index()

    # Region names
    region_names = list(data_wide.columns.drop("time"))

    return data_wide, region_names


def select_data(data_wide, region_names):
    data = data_wide[region_names].to_numpy()
    std_devs = np.std(data_wide[region_names], axis=0)

    return data, std_devs


def create_nodes_for_plotting(names):
    """
    Create node position arrays for the given AR6 region names.

    Parameters
    ----------
    names : list of str
        List of AR6 region abbreviations.

    Returns
    -------
    node_pos : dict
        Dictionary with 'x', 'y', and 'transform' for plotting.
    """
    # Define AR6 region positions
    ar6 = regionmask.defined_regions.ar6.all
    centroids = ar6.centroids

    # Map AR6 abbreviations to their (lon, lat) centroids
    pos_dict = {
        abbrev: (float(lon), float(lat))
        for abbrev, (lon, lat) in zip(ar6.abbrevs, centroids)
    }

    # Identify missing and valid names
    missing = [r for r in names if r not in pos_dict]
    if missing:
        print(f"⚠️ Warning: missing positions for {missing}. These will be skipped.")

    valid_names = [r for r in names if r in pos_dict]

    # Create node position arrays
    node_pos = {
        "x": np.array([pos_dict[r][0] for r in valid_names]),
        "y": np.array([pos_dict[r][1] for r in valid_names]),
        "transform": ccrs.PlateCarree(),
    }

    print(f"✅ node_pos created for {len(valid_names)} nodes.")
    return node_pos
