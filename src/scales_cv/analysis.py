from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import regionmask
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.models import LinearMediation
from tigramite.pcmci import PCMCI


def run_pcmci_analysis(
    data,
    region_names,
    tau_max=5,
    alpha_level=0.05,
    ci_test=ParCorr(),
    mask=None,
    mask_type=None,
    tag=None,
    plot_timeseries=False,
    output_folder=None,
    in_file=None,
    sourceDD=None,
    experiment_dd=None,
    node_pos=None,
    verbosity=1,
    show_figure=True,
):
    """
    Run PCMCI+ causal discovery and mediation analysis to obtain causal links and coefficients from data.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (time, region).
    region_names : list
        List of region names corresponding to columns in `data`.
    tau_max : int
        Maximum time lag to test.
    alpha_level : float, optional
        Significance level for PCMCI tests (default=0.05).
    ci_test : class, optional
        Conditional independence test (default=ParCorr).
    tag: string, optional
        Can be added to filename of network figure, for example "training" or "emulation".
    plot_timeseries : bool, optional
        Whether to show time series plots for each region.
    output_folder : Path or str, optional
        Folder to save output PNG (if None, figure is not saved).
    in_file : Path, optional
        Input file (used for naming output file).
    sourceDD : str, optional
        Model identifier, used in output filename.
    experiment_dd : str, optional
        Experiment identifier, used in output filename.
    node_pos : dict, optional
        Node positions for network plot.
    verbosity : int, optional
        Verbosity level for PCMCI (0 = silent, 1 = basic info, 2 = detailed).
    show_figure : bool
        Displays figure if True, otherwise not

    Returns
    -------
    results : dict
        PCMCI+ results.
    graph : np.ndarray
        Adjacency matrix of discovered links.
    Links : np.ndarray
        Val matrix (link strengths) from LinearMediation model.
    fig : matplotlib.figure.Figure
        The generated figure object (network plot).
    """
    # Create Tigramite DataFrame
    dataframe = pp.DataFrame(
        data, datatime={0: np.arange(len(data))}, mask=mask, var_names=region_names
    )

    if plot_timeseries:
        tp.plot_timeseries(
            dataframe,
            grey_masked_samples="data",
            show_meanline=True,
            figsize=(10, len(region_names) / 2),
        )

    # Run PCMCI+
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ci_test(mask_type=mask_type),
        verbosity=verbosity,
    )

    results = pcmci.run_pcmciplus(
        tau_min=1,
        tau_max=tau_max,
        pc_alpha=alpha_level,
        fdr_method="fdr_bh",
        link_assumptions=None,
        reset_lagged_links=True,
    )

    graph = pcmci.get_graph_from_pmatrix(
        p_matrix=results["p_matrix"],
        alpha_level=alpha_level,
        tau_min=1,
        tau_max=tau_max,
    )

    all_parents = pcmci.return_parents_dict(
        graph, val_matrix=results["val_matrix"], include_lagzero_parents=True
    )

    # Causal inference of link strengths
    med = LinearMediation(dataframe=dataframe, mask_type=mask_type)
    med.fit_model(all_parents=all_parents, tau_max=tau_max)
    Links = med.get_val_matrix()

    # Plot network on map
    fig, ax = plt.subplots(
        figsize=(12, 12), subplot_kw=dict(projection=ccrs.PlateCarree())
    )
    ax.add_feature(cfeature.COASTLINE, alpha=0.2)

    ar6 = regionmask.defined_regions.ar6.all

    ar6.plot_regions(
        ax=ax, add_label=False, line_kws=dict(color="k", linewidth=0.7, alpha=0.2)
    )

    tp.plot_graph(
        fig_ax=(fig, ax),
        val_matrix=Links,
        graph=graph,
        node_label_size=11,
        node_size=11,
        link_label_fontsize=13,
        var_names=region_names,
        show_colorbar=False,
        node_pos=node_pos,
    )

    # Save figure if requested
    if output_folder and in_file:
        in_file_name = in_file.stem.replace("_", "-")
        outfile = Path(output_folder) / (
            f"{in_file_name}_{sourceDD}_{experiment_dd}_"
            f"{ci_test.__name__}_tmin-1_tmax-{tau_max}_alpha-{alpha_level}_"
            f"{len(region_names)}_{tag}_network.png"
        )
        fig.savefig(outfile, dpi=150, bbox_inches="tight")

    # Optionally display figure
    if show_figure:
        plt.show()
    else:
        plt.close(fig)  # prevent it from being displayed in notebooks

    return results, graph, Links, fig


def define_mask(data, selection=None):
    """
    Create a mask for a 2D array where rows correspond to months.
    1 = masked, 0 = included.

    Parameters
    ----------
        data (np.ndarray): 2D array with shape (time, features)
        selection (str or list of str): Either a season ('DJF', 'MAM', 'JJA', 'SON')
                                        or single month(s) like 'Jan', 'Feb', etc.

    Returns
    -------
        np.ndarray: Mask with same shape as `data` (1=masked, 0=included)
    """
    n_rows, n_cols = data.shape

    # Month-to-index mapping
    month_to_idx = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11,
    }

    # Seasonal patterns: 0 = selected, 1 = masked
    seasonal_patterns = {
        "DJF": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        "MAM": [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        "JJA": [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        "SON": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    }

    # If selection is a season
    if isinstance(selection, str) and selection.upper() in seasonal_patterns:
        pattern = seasonal_patterns[selection.upper()]

    # If selection is month(s)
    else:
        if isinstance(selection, str):
            selection = [selection]  # make it a list for uniform processing
        pattern = [0 if month in selection else 1 for month in month_to_idx.keys()]

    # Repeat pattern along rows
    repeats = int(np.ceil(n_rows / 12))
    full_pattern = np.tile(pattern, repeats)[:n_rows]

    # Broadcast along columns
    mask = np.tile(full_pattern[:, None], (1, n_cols))

    return mask
