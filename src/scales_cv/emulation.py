import numpy as np
import pandas as pd


def val_matrix_to_links_dict(val_matrix, tau_min=1, tau_max=None, tol=1e-12):
    """
    Convert a Tigramite val_matrix (shape N, N, tau_max+1)
    into a dictionary of link coefficients like:
        {j: [((i, -tau), coeff), ...], ...}
    where coeff != 0 (within tolerance).
    """
    N, _, Tmax = val_matrix.shape
    if tau_max is None:
        tau_max = Tmax - 1

    links_coeffs = {}
    for j in range(N):  # target variable
        links = []
        for i in range(N):  # source variable
            for tau in range(tau_min, tau_max + 1):
                coeff = val_matrix[j, i, tau]
                if abs(coeff) > tol:  # skip exact zeros / numerical noise
                    links.append(((i, -tau), float(coeff)))
        if links:
            links_coeffs[j] = links

    if len(val_matrix) - 1 not in links_coeffs.keys():
        links_coeffs[len(val_matrix) - 1] = [
            ((len(val_matrix) - 1, -1), np.float64(0.0))
        ]

    return links_coeffs


def rescale_links_coeffs(links_coeffs, sigma_orig):
    """
    Rescale VAR coefficients from standardized data to original units.

    Parameters
    ----------
    links_coeffs : dict
        Dictionary of VAR coefficients: {j: [((i, -tau), coeff), ...], ...}
        where j = parent node, i = target node, tau = lag.
    sigma_orig : array-like, shape (N,)
        Standard deviation of each variable in the target/original units.

    Returns
    -------
    links_rescaled : dict
        Rescaled coefficients in the same format as input.
    """
    sigma_orig = np.array(sigma_orig)
    links_rescaled = {}
    for j, lst in links_coeffs.items():
        links_rescaled[j] = []
        for (i, tau), coeff in lst:
            coeff_rescaled = coeff * sigma_orig[i] / sigma_orig[j]
            links_rescaled[j].append(((i, tau), coeff_rescaled))
    return links_rescaled


def build_matrices(links_dict, n_vars=None, tau_max=None, verbose=False):
    """
    Build VAR coefficient matrices A_1, A_2, ..., A_p from a PCMCI+-style dictionary.

    Parameters
    ----------
    links_dict : dict
        Dictionary mapping each parent variable index to a list of ((target, -lag), coefficient) tuples.
        Example:
            {
                0: [((0, -1), 0.6), ((1, -1), 0.3)],
                1: [((2, -2), -0.2)],
                ...
            }
        Means: var0(t-1)→var0(t), var0(t-1)→var1(t), var1(t-2)→var2(t)
    n_vars : int, optional
        Number of variables (if None, inferred from dict keys and entries)
    tau_max : int, optional
        Maximum lag (if None, inferred from the largest |lag| found)
    verbose : bool, optional
        Print summary of all filled entries

    Returns
    -------
    A_matrices : list of np.ndarray
        List [A1, A2, ..., A_tau_max], each of shape (n_vars, n_vars),
        where A_lag[i,j] = effect of variable j (at t−lag) on variable i (at time t)
    """
    # Infer number of variables and max lag if not provided
    all_vars = set(links_dict.keys())
    max_lag_found = 0
    for src, targets in links_dict.items():
        all_vars.add(src)
        for (tgt, lag_tuple), coeff in targets:
            all_vars.add(tgt)
            max_lag_found = max(max_lag_found, abs(lag_tuple))

    if n_vars is None:
        n_vars = max(all_vars) + 1
    if tau_max is None:
        tau_max = max_lag_found

    # Initialize list of A matrices
    A_matrices = [np.zeros((n_vars, n_vars), dtype=float) for _ in range(tau_max)]

    # Fill coefficients
    for parent, links in links_dict.items():
        for (target, lag_neg), coeff in links:
            lag = abs(lag_neg)
            if 1 <= lag <= tau_max:
                A_matrices[lag - 1][target, parent] = float(coeff)
                if verbose:
                    print(f"Var{parent}(t-{lag}) -> Var{target}(t): {coeff:.6f}")
            elif verbose:
                print(f"Skipping lag {lag} (outside tau_max={tau_max})")

    return A_matrices


def simulate_VAR(A_matrices, std_devs, T=1000, burn=200, verbose=True, seed=None):
    """
    Simulate a VAR(p) process with specified per-variable standard deviations.

    Parameters
    ----------
    A_matrices : list of np.ndarray
        List of VAR coefficient matrices [A1, A2, ..., Ap], each shape (N, N)
    std_devs : array_like
        Standard deviations of the noise for each variable (length N)
    T : int
        Number of time steps to simulate after burn-in
    burn : int
        Number of initial samples to discard
    verbose : bool
        Print stationarity info
    seed : int
        Random seed

    Returns
    -------
    Y_sim : np.ndarray
        Simulated time series of shape (T, N)
    eigvals : np.ndarray
        Eigenvalues of the companion matrix
    """
    if seed is not None:
        np.random.seed(seed)

    N = A_matrices[0].shape[0]
    p = len(A_matrices)
    total_steps = T + burn

    # Build companion matrix F for stationarity check
    F = np.zeros((N * p, N * p))
    for i, A in enumerate(A_matrices):
        F[:N, i * N : (i + 1) * N] = A
    if p > 1:
        F[N:, :-N] = np.eye(N * (p - 1))

    eigvals = np.linalg.eigvals(F)
    max_abs_eig = np.max(np.abs(eigvals))
    if verbose:
        print(f"Max |eig(F)| = {max_abs_eig:.4f}")
        if max_abs_eig < 1:
            print("System is stationary (all eigenvalues < 1)")
        else:
            print("WARNING: System may not be stationary!")

    # Simulate VAR
    Y = np.zeros((total_steps, N))
    # initialize first p steps
    for i in range(p):
        Y[i] = np.random.randn(N) * std_devs

    for t in range(p, total_steps):
        lagged_sum = np.zeros(N)
        for lag, A in enumerate(A_matrices, start=1):
            lagged_sum += A @ Y[t - lag]
        noise = np.random.randn(N) * std_devs
        Y[t] = lagged_sum + noise

    Y_sim = Y[burn:, :]

    if verbose:
        print(
            f"✅ Simulation complete: {Y_sim.shape[0]} samples, {N} variables, {p} lags"
        )

    return Y_sim, eigvals


def build_seasonal_A_matrices(season_links_dict, n_vars=None, tau_max=None):
    """
    season_links_dict: dict mapping season_key -> Links_dict (PCMCI style)
    returns: dict season_key -> list of A_matrices (A1...Ap)
    """
    seasonal_A = {}
    # infer global tau_max if not given: choose max across seasons
    inferred_tau = 0
    for links in season_links_dict.values():
        # inspect for lag in links
        for parent, links_list in links.items():
            for (tgt, lag_neg), coeff in links_list:
                inferred_tau = max(inferred_tau, abs(lag_neg))
    if tau_max is None:
        tau_max = inferred_tau

    # build each season's A_matrices and pad/truncate to tau_max
    for season, links in season_links_dict.items():
        A_list = build_matrices(links, n_vars=n_vars, tau_max=tau_max, verbose=False)
        seasonal_A[season] = A_list  # already length tau_max
    return seasonal_A


def simulate_seasonal_VAR(
    seasonal_A, std_devs, season_seq, T=1000, burn=200, seed=None, verbose=True
):
    """
    seasonal_A: dict season_key -> list of A_matrices for that season
    std_devs: global target standard deviations (one per variable)
    season_seq: list/array of length (T+burn): season for each timestep
    """
    if seed is not None:
        np.random.seed(seed)

    season_keys = list(seasonal_A.keys())
    p_list = [len(Alist) for Alist in seasonal_A.values()]
    if len(set(p_list)) != 1:
        raise ValueError("All seasons must have same number of lags p.")

    p = p_list[0]
    N = seasonal_A[season_keys[0]][0].shape[0]

    total_steps = T + burn
    if len(season_seq) < total_steps:
        raise ValueError("season_seq must be at least T+burn long")

    sigma_Y2 = np.array(std_devs) ** 2  # global target variances

    # Compute per-season innovation stds but using GLOBAL target std_devs
    seasonal_sigma_eps = {}
    for season, Alist in seasonal_A.items():
        sigma_eps2 = sigma_Y2.copy()
        for i in range(N):
            variance_contrib = 0.0
            for A in Alist:
                variance_contrib += np.sum(A[i, :] ** 2 * sigma_Y2)
            sigma_eps2[i] = sigma_Y2[i] - variance_contrib

        sigma_eps2 = np.clip(sigma_eps2, 1e-8, None)
        seasonal_sigma_eps[season] = np.sqrt(sigma_eps2)

    # Simulate
    Y = np.zeros((total_steps, N))

    # initialize using the first season's noise
    init_season = season_seq[0]
    init_std = seasonal_sigma_eps[init_season]
    for t in range(p):
        Y[t] = np.random.randn(N) * init_std

    for t in range(p, total_steps):
        season = season_seq[t]
        Alist = seasonal_A[season]
        eps_std = seasonal_sigma_eps[season]

        lag_sum = np.zeros(N)
        for lag, A in enumerate(Alist, start=1):
            lag_sum += A @ Y[t - lag]

        noise = np.random.randn(N) * eps_std
        Y[t] = lag_sum + noise

    Y_sim = Y[burn:, :]

    if verbose:
        print(f"Seasonal VAR simulation complete: {T} samples, p={p}, N={N}")

    return Y_sim


def causal_model(
    Links, tau_max, std_devs, names, T=500, burn=200, seed=None, verbose=True
):
    """
    Build a VAR system from link coefficients and simulate emulated time series,
    automatically computing innovation stds to match target stds analytically.

    Parameters
    ----------
    Links : dict
        PCMCI+-style dictionary of links
    tau_max : int
        Maximum time lag
    std_devs : array-like
        Target standard deviations for each variable
    names : list of str
        Variable names
    T : int
        Length of simulated series after burn-in
    burn : int
        Burn-in length
    seed : int
        Random seed
    verbose : bool
        Print debug info

    Returns
    -------
    Y_sim : np.ndarray
        Simulated time series (T x N)
    df : pd.DataFrame
        Simulated time series as DataFrame
    em_std_devs : np.ndarray
        Standard deviations of simulated series
    coeffs : dict
        Rescaled link coefficients used
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert link matrix to dict and sort
    links_coeffs = val_matrix_to_links_dict(Links, tau_max=tau_max)
    links_coeffs = dict(sorted(links_coeffs.items()))

    # Rescale coefficients by original std devs
    coeffs = rescale_links_coeffs(links_coeffs, np.array(std_devs))

    # Build VAR coefficient matrices
    A_matrices = build_matrices(coeffs, tau_max=tau_max)

    # Analytic noise std dev correction
    N = len(std_devs)
    sigma_Y2 = np.array(std_devs) ** 2  # target variances
    sigma_eps2 = sigma_Y2.copy()

    for i in range(N):
        variance_contrib = 0.0
        for A in A_matrices:
            # sum of squares of incoming links multiplied by target variances
            variance_contrib += np.sum(A[i, :] ** 2 * sigma_Y2)
        sigma_eps2[i] = sigma_Y2[i] - variance_contrib

    # Check for negative values
    if np.any(sigma_eps2 <= 0):
        if verbose:
            print("Warning: some innovation variances <= 0 due to strong coupling.")
            print("Clipping to small positive number (1e-6)")
        sigma_eps2 = np.clip(sigma_eps2, 1e-6, None)

    sigma_eps = np.sqrt(sigma_eps2)

    # Simulate VAR
    Y_sim, eigvals = simulate_VAR(
        A_matrices, sigma_eps, T=T, burn=burn, verbose=verbose, seed=seed
    )

    df = pd.DataFrame(Y_sim, columns=names)
    em_std_devs = df.std(axis=0).values

    return Y_sim, df, em_std_devs, coeffs


def compare_dicts(coeffs, em_coeffs, names, run_id, tolerance=0.1):
    """
    Compare training (coeffs) and emulated (em_coeffs) causal link dictionaries.

    Returns
    -------
      - df_diff: DataFrame with differences (missing, hallucinated, sign-change, strength)
      - df_all_links: DataFrame with all detected links (input_data & emulation)
    """
    result = {
        "missing_links": {},
        "hallucinated_links": {},
        "sign_differences": {},
        "value_differences": {},
    }

    # Step 1: Compare structure of training and emulated link dictionaries
    for key in coeffs:
        list1 = coeffs[key]
        list2 = em_coeffs.get(key, [])

        dict1_pairs = {pair: value for pair, value in list1}
        dict2_pairs = {pair: value for pair, value in list2}

        # Missing in emulated
        for pair in dict1_pairs:
            if pair not in dict2_pairs:
                result["missing_links"].setdefault(key, []).append(pair)

        # Hallucinated (present in emulated, not in training)
        for pair in dict2_pairs:
            if pair not in dict1_pairs:
                result["hallucinated_links"].setdefault(key, []).append(pair)

        # Compare overlapping pairs
        for pair in dict1_pairs:
            if pair in dict2_pairs:
                val1 = dict1_pairs[pair]
                val2 = dict2_pairs[pair]

                # Sign difference
                if (val1 >= 0 and val2 < 0) or (val1 < 0 and val2 >= 0):
                    result["sign_differences"].setdefault(key, {})[pair] = (val1, val2)
                else:
                    diff = val1 - val2
                    if abs(diff) > tolerance:
                        result["value_differences"].setdefault(key, {})[pair] = diff

    # Step 2: Build DataFrame of differences
    records = []

    def add_row(run_id, type_, parent, target, lag, training, emulated):
        delta = np.nan
        if training is not None and emulated is not None:
            delta = training - emulated
        records.append(
            {
                "run_id": run_id,
                "type": type_,
                "parent": parent,
                "target": target,
                "time-lag": lag,
                "training": training,
                "emulated": emulated,
                "delta": delta,
            }
        )

    def get_coeff(dictionary, parent, pair):
        for p, v in dictionary.get(parent, []):
            if p == pair:
                return v
        return None

    # Missing links
    for parent, pairs in result["missing_links"].items():
        for target, lag in pairs:
            add_row(
                run_id,
                "missing",
                names[parent],
                names[target],
                lag,
                get_coeff(coeffs, parent, (target, lag)),
                None,
            )

    # Hallucinated links
    for parent, pairs in result["hallucinated_links"].items():
        for target, lag in pairs:
            add_row(
                run_id,
                "hallucinated",
                names[parent],
                names[target],
                lag,
                None,
                get_coeff(em_coeffs, parent, (target, lag)),
            )

    # Sign changes
    for parent, pairs in result["sign_differences"].items():
        for (target, lag), (v1, v2) in pairs.items():
            add_row(run_id, "sign-change", names[parent], names[target], lag, v1, v2)

    # Value differences
    for parent, pairs in result["value_differences"].items():
        for (target, lag), diff in pairs.items():
            v1 = get_coeff(coeffs, parent, (target, lag))
            v2 = get_coeff(em_coeffs, parent, (target, lag))
            add_row(run_id, "strength", names[parent], names[target], lag, v1, v2)

    df_diff = pd.DataFrame(records)

    # Step 3: Build DataFrame of all links (input + emulation)
    def flatten_links(links_dict, source_type):
        data = []
        for parent, lst in links_dict.items():
            for (target, lag), strength in lst:
                data.append(
                    {
                        "run_id": run_id,
                        "source": source_type,
                        "parent": names[parent],
                        "target": names[target],
                        "time-lag": lag,
                        "strength": float(strength),
                    }
                )
        return pd.DataFrame(data)

    df_input_all = flatten_links(coeffs, "input_data")
    df_emul_all = flatten_links(em_coeffs, "emulation")
    df_all_links = pd.concat([df_input_all, df_emul_all], ignore_index=True)

    # Step 4: Return both
    return df_diff, df_all_links


def aggregate_links_weighted(
    Links_dict,
    min_freq_to_keep=8,  # minimum number of members that must have a nonzero for that position
    method="mean",  # 'mean' | 'median'
    weight_mode="freq/num",  # 'freq/num' | 'freq/req' | 'power'
    gamma=1.0,  # used if weight_mode=='power'
    required_count=None,  # used if weight_mode=='freq/req'; defaults to min_freq_to_keep
    num_members=None,  # inferred if None
    return_freq=True,
):
    """
    Aggregate (p,N,M) arrays in Links_dict into one set of p matrices with proportion-weighting.

    Returns
    -------
    aggregated : np.array shape (p, N, M)
    freq_per_lag : np.array shape (p, N, M)  (only if return_freq True)
    """
    if num_members is None:
        num_members = len(Links_dict)
    if required_count is None:
        required_count = min_freq_to_keep

    # determine dimensions
    example = next(iter(Links_dict))
    arr0 = Links_dict[example]  # shape (p, N, M)
    p, N, M = arr0.shape

    aggregated = np.zeros((p, N, M))
    freq_per_lag = np.zeros((p, N, M), dtype=int)

    def base_stat(vals):
        if method == "mean":
            return np.mean(vals)
        elif method == "median":
            return np.median(vals)
        else:
            raise ValueError("method must be 'mean' or 'median'")

    for lag in range(p):
        stacked = np.stack(
            [Links_dict[k][lag] for k in Links_dict]
        )  # shape (num_members, N, M)
        nonzero_mask = stacked != 0
        freq = nonzero_mask.sum(axis=0)  # shape (N, M)
        freq_per_lag[lag] = freq

        keep_mask = freq >= min_freq_to_keep
        if not np.any(keep_mask):
            continue

        for r, c in np.argwhere(keep_mask):
            vals = stacked[:, r, c]
            nonzeros = vals[vals != 0]
            if nonzeros.size == 0:
                continue

            base = base_stat(nonzeros)

            # compute weight
            if weight_mode == "freq/num":
                weight = freq[r, c] / float(num_members)
            elif weight_mode == "freq/req":
                weight = min(1.0, freq[r, c] / float(required_count))
            elif weight_mode == "power":
                weight = (freq[r, c] / float(num_members)) ** float(gamma)
            else:
                raise ValueError("weight_mode must be 'freq/num','freq/req',or 'power'")

            aggregated[lag, r, c] = base * weight

    return aggregated, freq_per_lag
