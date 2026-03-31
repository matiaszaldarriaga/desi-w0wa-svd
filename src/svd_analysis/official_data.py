"""
Official DESI DR2 and supernova data loading.

This module loads paper-grade data from official sources:
- DESI DR2 BAO: CobayaSampler/bao_data repository
- Supernova: Pantheon+, DES Y5, Union3 from CobayaSampler/sn_data

The data uses the official {DM/rs, DH/rs} parameterization with full
covariance matrices including off-diagonal correlations.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Literal

import numpy as np
from numpy.typing import NDArray


class BAOObservable(TypedDict):
    """Single BAO observable."""
    z: float
    value: float
    quantity: Literal['DV_over_rs', 'DM_over_rs', 'DH_over_rs']
    tracer: str


class OfficialBAOData(TypedDict):
    """Official DESI DR2 BAO data structure."""
    observables: list[BAOObservable]
    data_vector: NDArray[np.floating]
    covariance: NDArray[np.floating]
    errors: NDArray[np.floating]
    correlation: NDArray[np.floating]
    whitening_matrix: NDArray[np.floating]
    whitened_data: NDArray[np.floating]


class SNData(TypedDict):
    """Supernova data structure."""
    z: NDArray[np.floating]
    mu: NDArray[np.floating]
    covariance: NDArray[np.floating]
    errors: NDArray[np.floating]


def get_data_path() -> Path:
    """Get path to official data directory."""
    # Navigate from this file (src/svd_analysis/) to data/
    src_dir = Path(__file__).parent
    project_root = src_dir.parent.parent
    return project_root / "data"


def _load_cobaya_covariance(cov_file: Path, expected_n: int) -> NDArray[np.floating]:
    """
    Load a covariance matrix in Cobaya format.

    Format: first line = integer N (dimension), then N*N values (one per line).

    Parameters
    ----------
    cov_file : Path
        Path to covariance file.
    expected_n : int
        Expected dimension (number of data points).

    Returns
    -------
    NDArray
        (N, N) covariance matrix.

    Raises
    ------
    ValueError
        If file dimension doesn't match expected_n or data count is wrong.
    """
    with open(cov_file) as f:
        # First line is the dimension
        n = int(f.readline().strip())
        if n != expected_n:
            raise ValueError(
                f"Covariance dimension {n} doesn't match expected {expected_n} "
                f"in {cov_file}"
            )
        # Read remaining N*N values
        values = []
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))

    if len(values) != n * n:
        raise ValueError(
            f"Expected {n*n} covariance values, got {len(values)} in {cov_file}"
        )

    return np.array(values).reshape(n, n)


def load_official_bao_data(
    data_dir: Path | None = None,
    dataset: str = "ALL_GCcomb"
) -> OfficialBAOData:
    """
    Load official DESI DR2 BAO data.

    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory. If None, uses default input/data path.
    dataset : str
        Dataset identifier. Options: "ALL_GCcomb" (default, combined),
        or individual tracers like "BGS_BRIGHT-21.35_GCcomb".

    Returns
    -------
    OfficialBAOData
        Dictionary containing:
        - observables: List of individual BAO observables
        - data_vector: 1D array of observable values
        - covariance: Full covariance matrix
        - errors: Diagonal errors (sqrt of covariance diagonal)
        - correlation: Correlation matrix
        - whitening_matrix: L^{-1} where C = L L^T (Cholesky)
        - whitened_data: Whitened data vector
    """
    if data_dir is None:
        data_dir = get_data_path()

    bao_dir = data_dir / "bao_data" / "desi_bao_dr2"

    mean_file = bao_dir / f"desi_gaussian_bao_{dataset}_mean.txt"
    cov_file = bao_dir / f"desi_gaussian_bao_{dataset}_cov.txt"

    if not mean_file.exists():
        raise FileNotFoundError(
            f"BAO data file not found: {mean_file}\n"
            f"Clone the data repository:\n"
            f"  git clone https://github.com/CobayaSampler/bao_data input/data/bao_data"
        )

    # Parse mean file
    observables: list[BAOObservable] = []
    with open(mean_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                z = float(parts[0])
                value = float(parts[1])
                quantity = parts[2]

                # Determine tracer from redshift
                tracer = _get_tracer_from_z(z, quantity)

                observables.append({
                    'z': z,
                    'value': value,
                    'quantity': quantity,  # type: ignore
                    'tracer': tracer
                })

    # Parse covariance matrix
    cov_lines = []
    with open(cov_file) as f:
        for line in f:
            if line.strip():
                cov_lines.append([float(x) for x in line.split()])
    covariance = np.array(cov_lines)

    # Build data vector
    data_vector = np.array([obs['value'] for obs in observables])

    # Compute derived quantities
    errors = np.sqrt(np.diag(covariance))

    # Correlation matrix
    d_inv = np.diag(1.0 / errors)
    correlation = d_inv @ covariance @ d_inv

    # Whitening matrix via Cholesky decomposition
    # C = L L^T, so C^{-1} = L^{-T} L^{-1}
    # Whitening: y = L^{-1} x has identity covariance
    L = np.linalg.cholesky(covariance)
    L_inv = np.linalg.inv(L)
    whitening_matrix = L_inv

    # Whitened data
    whitened_data = whitening_matrix @ data_vector

    return {
        'observables': observables,
        'data_vector': data_vector,
        'covariance': covariance,
        'errors': errors,
        'correlation': correlation,
        'whitening_matrix': whitening_matrix,
        'whitened_data': whitened_data
    }


def _get_tracer_from_z(z: float, quantity: str) -> str:
    """Determine tracer name from redshift."""
    if abs(z - 0.295) < 0.01:
        return "BGS"
    elif abs(z - 0.51) < 0.01:
        return "LRG1"
    elif abs(z - 0.706) < 0.01:
        return "LRG2"
    elif abs(z - 0.934) < 0.01:
        return "LRG3+ELG1"
    elif abs(z - 1.321) < 0.01:
        return "ELG2"
    elif abs(z - 1.484) < 0.01:
        return "QSO"
    elif abs(z - 2.33) < 0.1:
        return "Lya"
    else:
        return "Unknown"


def load_pantheon_plus(data_dir: Path | None = None) -> SNData:
    """
    Load Pantheon+ SH0ES supernova data.

    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory.

    Returns
    -------
    SNData
        Dictionary containing z, mu, covariance, errors.
    """
    if data_dir is None:
        data_dir = get_data_path()

    sn_dir = data_dir / "sn_data" / "PantheonPlus"
    data_file = sn_dir / "Pantheon+SH0ES.dat"
    cov_file = sn_dir / "Pantheon+SH0ES_STAT+SYS.cov"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Pantheon+ data not found: {data_file}\n"
            f"Clone the data repository:\n"
            f"  git clone https://github.com/CobayaSampler/sn_data input/data/sn_data"
        )

    # Parse data file (space-separated with header)
    z_list = []
    mu_list = []

    with open(data_file) as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.split()
            if len(parts) > 10:
                z_hd = float(parts[2])  # zHD column
                mu = float(parts[10])   # MU_SH0ES column
                z_list.append(z_hd)
                mu_list.append(mu)

    z = np.array(z_list)
    mu = np.array(mu_list)

    # Load covariance
    if cov_file.exists():
        covariance = _load_cobaya_covariance(cov_file, len(z))
    else:
        covariance = np.diag(np.ones(len(z)) * 0.1**2)

    errors = np.sqrt(np.diag(covariance))

    return {
        'z': z,
        'mu': mu,
        'covariance': covariance,
        'errors': errors
    }


def load_des_y5(data_dir: Path | None = None) -> SNData:
    """
    Load DES Y5 supernova data.

    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory.

    Returns
    -------
    SNData
        Dictionary containing z, mu, covariance, errors.
    """
    if data_dir is None:
        data_dir = get_data_path()

    sn_dir = data_dir / "sn_data" / "DESY5"
    data_file = sn_dir / "DES-SN5YR_HD.csv"

    if not data_file.exists():
        raise FileNotFoundError(
            f"DES Y5 data not found: {data_file}\n"
            f"Clone the data repository:\n"
            f"  git clone https://github.com/CobayaSampler/sn_data input/data/sn_data"
        )

    # Parse CSV — load ALL entries first (covariance is N_total × N_total)
    z_list = []
    mu_list = []
    err_list = []

    with open(data_file) as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                z_cmb = float(parts[2])  # zCMB
                mu = float(parts[5])     # MU
                err = float(parts[6])    # MUERR_FINAL
                z_list.append(z_cmb)
                mu_list.append(mu)
                err_list.append(err)

    z_all = np.array(z_list)
    mu_all = np.array(mu_list)
    errors_all = np.array(err_list)

    # Load full covariance from covsys_000.txt if available
    cov_file = sn_dir / "covsys_000.txt"
    if cov_file.exists():
        covariance_all = _load_cobaya_covariance(cov_file, len(z_all))
    else:
        covariance_all = np.diag(errors_all**2)

    # Filter outliers with huge errors, keeping covariance consistent
    good = errors_all < 100
    z = z_all[good]
    mu = mu_all[good]
    covariance = covariance_all[np.ix_(good, good)]
    errors = np.sqrt(np.diag(covariance))

    return {
        'z': z,
        'mu': mu,
        'covariance': covariance,
        'errors': errors
    }


def load_des_dovekie(data_dir: Path | None = None) -> SNData:
    """
    Load DES-Dovekie supernova data (Popovic et al. 2025, arXiv:2511.07517).

    This is a recalibrated version of DES Y5 with corrected calibration,
    F99 color law fix, and retrained SALT3 model. The covariance is provided
    as an inverse (precision matrix) in npz format.

    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory.

    Returns
    -------
    SNData
        Dictionary containing z, mu, covariance, errors.
    """
    if data_dir is None:
        data_dir = get_data_path()

    sn_dir = data_dir / "sn_data" / "DES-Dovekie"
    data_file = sn_dir / "DES-Dovekie_HD.csv"

    if not data_file.exists():
        raise FileNotFoundError(
            f"DES-Dovekie data not found: {data_file}\n"
            f"Clone the data repository:\n"
            f"  git clone https://github.com/CobayaSampler/sn_data input/data/sn_data"
        )

    # Parse CSV
    z_list = []
    mu_list = []
    err_list = []

    with open(data_file) as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 9:
                z_hd = float(parts[2])    # zHD
                mu = float(parts[4])      # MU
                err = float(parts[5])     # MUERR
                z_list.append(z_hd)
                mu_list.append(mu)
                err_list.append(err)

    z_all = np.array(z_list)
    mu_all = np.array(mu_list)
    errors_all = np.array(err_list)

    # Load inverse covariance from npz (packed upper triangular)
    cov_inv_file = sn_dir / "covtot_inv_000.npz"
    if cov_inv_file.exists():
        npz = np.load(cov_inv_file)
        nsn = int(npz['nsn'][0])
        cov_flat = npz['cov'].astype(np.float64)

        # Reconstruct full symmetric matrix from upper triangular
        cov_inv = np.zeros((nsn, nsn))
        idx = 0
        for i in range(nsn):
            for j in range(i, nsn):
                cov_inv[i, j] = cov_flat[idx]
                cov_inv[j, i] = cov_inv[i, j]
                idx += 1

        # Invert to get covariance
        covariance_all = np.linalg.inv(cov_inv)
    else:
        covariance_all = np.diag(errors_all**2)

    # Filter outliers with huge errors (same threshold as DES Y5)
    good = errors_all < 100
    z = z_all[good]
    mu = mu_all[good]
    covariance = covariance_all[np.ix_(good, good)]
    errors = np.sqrt(np.diag(covariance))

    return {
        'z': z,
        'mu': mu,
        'covariance': covariance,
        'errors': errors
    }


def load_union3(data_dir: Path | None = None) -> SNData:
    """
    Load Union3 binned supernova data.

    Parameters
    ----------
    data_dir : Path, optional
        Path to data directory.

    Returns
    -------
    SNData
        Dictionary containing z, mu (actually mb), covariance, errors.
    """
    if data_dir is None:
        data_dir = get_data_path()

    sn_dir = data_dir / "sn_data" / "Union3"
    data_file = sn_dir / "lcparam_full.txt"
    cov_file = sn_dir / "mag_covmat.txt"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Union3 data not found: {data_file}\n"
            f"Clone the data repository:\n"
            f"  git clone https://github.com/CobayaSampler/sn_data input/data/sn_data"
        )

    # Parse data file
    z_list = []
    mb_list = []

    with open(data_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                z_cmb = float(parts[1])  # zcmb
                mb = float(parts[4])     # mb
                z_list.append(z_cmb)
                mb_list.append(mb)

    z = np.array(z_list)
    mu = np.array(mb_list)  # Note: this is mb, not distance modulus

    # Load covariance
    if cov_file.exists():
        covariance = _load_cobaya_covariance(cov_file, len(z))
    else:
        covariance = np.diag(np.ones(len(z)) * 0.1**2)

    errors = np.sqrt(np.diag(covariance))

    return {
        'z': z,
        'mu': mu,
        'covariance': covariance,
        'errors': errors
    }


def print_bao_summary(data: OfficialBAOData) -> None:
    """Print summary of BAO data."""
    print("=" * 70)
    print("OFFICIAL DESI DR2 BAO DATA")
    print("=" * 70)
    print(f"\nNumber of observables: {len(data['observables'])}")
    print(f"Data vector shape: {data['data_vector'].shape}")
    print(f"Covariance shape: {data['covariance'].shape}")

    print("\nObservables:")
    print(f"{'#':>3} {'z':>8} {'Value':>12} {'Error':>10} {'Quantity':>12} {'Tracer':>10}")
    print("-" * 60)
    for i, obs in enumerate(data['observables']):
        err = data['errors'][i]
        print(f"{i+1:>3} {obs['z']:>8.3f} {obs['value']:>12.4f} {err:>10.4f} "
              f"{obs['quantity']:>12} {obs['tracer']:>10}")

    print("\nCorrelations (off-diagonal, |r| > 0.1):")
    n = len(data['observables'])
    for i in range(n):
        for j in range(i+1, n):
            r = data['correlation'][i, j]
            if abs(r) > 0.1:
                print(f"  Obs {i+1}-{j+1}: r = {r:+.3f}")


class BinnedSNData(TypedDict):
    """Binned supernova data structure."""
    z: NDArray[np.floating]
    mu: NDArray[np.floating]
    covariance: NDArray[np.floating]
    errors: NDArray[np.floating]
    n_per_bin: NDArray[np.signedinteger]


def get_union3_bin_grid(data_dir: Path | None = None) -> NDArray[np.floating]:
    """
    Return Union3 redshift bin centers as the common binning grid.

    These are the 22 redshifts from the Union3 dataset, used as the
    standard grid for cross-dataset comparison.
    """
    u3 = load_union3(data_dir)
    return u3['z'].copy()


def bin_sn_data(
    z: NDArray[np.floating],
    mu: NDArray[np.floating],
    covariance: NDArray[np.floating],
    bin_centers: NDArray[np.floating],
) -> BinnedSNData:
    """
    Bin SN data to specified redshift grid with full covariance propagation.

    Constructs a linear binning matrix B such that:
        mu_bin = B @ mu
        cov_bin = B @ covariance @ B.T

    Within each bin, SNe are combined with inverse-variance weighting
    using diagonal covariance elements: w_i = 1/C[i,i].

    Parameters
    ----------
    z : (N,) redshifts of individual SNe
    mu : (N,) distance moduli
    covariance : (N, N) covariance matrix
    bin_centers : (N_bin,) target bin center redshifts

    Returns
    -------
    BinnedSNData
        z: (N_bin,) weighted-mean redshifts within each bin
        mu: (N_bin,) binned distance moduli
        covariance: (N_bin, N_bin) propagated covariance
        errors: (N_bin,) diagonal errors
        n_per_bin: (N_bin,) number of SNe per bin
    """
    n_sn = len(z)
    n_bins = len(bin_centers)

    # Bin edges from midpoints of bin_centers
    bin_edges = np.zeros(n_bins + 1)
    bin_edges[0] = max(0, bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2)
    bin_edges[-1] = bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2
    for i in range(1, n_bins):
        bin_edges[i] = (bin_centers[i - 1] + bin_centers[i]) / 2

    # Inverse-variance weights from diagonal
    diag_var = np.diag(covariance)
    w = np.where(diag_var > 0, 1.0 / diag_var, 0.0)

    # Build binning matrix B: (n_bins, n_sn)
    B = np.zeros((n_bins, n_sn))
    z_bin = np.zeros(n_bins)
    n_per_bin = np.zeros(n_bins, dtype=int)

    for j in range(n_bins):
        mask = (z >= bin_edges[j]) & (z < bin_edges[j + 1])
        n_in_bin = np.sum(mask)
        n_per_bin[j] = n_in_bin

        if n_in_bin > 0:
            w_bin = w[mask]
            W_j = np.sum(w_bin)
            if W_j > 0:
                B[j, mask] = w_bin / W_j
                z_bin[j] = np.average(z[mask], weights=w_bin)
            else:
                # All zero weights — use equal weighting
                B[j, mask] = 1.0 / n_in_bin
                z_bin[j] = np.mean(z[mask])
        else:
            # Empty bin: keep bin_center, B row stays zero
            z_bin[j] = bin_centers[j]

    # Binned distance moduli
    mu_bin = B @ mu

    # Propagate covariance
    cov_bin = B @ covariance @ B.T

    # For empty bins: set large variance (no information)
    for j in range(n_bins):
        if n_per_bin[j] == 0:
            cov_bin[j, j] = 1e6

    errors = np.sqrt(np.diag(cov_bin))

    return {
        'z': z_bin,
        'mu': mu_bin,
        'covariance': cov_bin,
        'errors': errors,
        'n_per_bin': n_per_bin,
        'B': B,
        'z_raw': z.copy(),
    }


def compute_effective_redshifts(
    ref_cosmo,
    z_bin: NDArray[np.floating],
    B: NDArray[np.floating],
    z_raw: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute effective redshifts that absorb the Jensen correction.

    Finds z_eff such that mu_ref(z_eff) = (B @ mu_ref(z_raw)) for each bin,
    ensuring consistency between binned data and model predictions. This
    removes the second-order bias from evaluating a nonlinear function at
    the weighted-mean redshift instead of properly averaging over the bin.

    Parameters
    ----------
    ref_cosmo : w0waCDM
        Reference cosmology (astropy).
    z_bin : (N_bin,)
        Weighted-mean redshifts of the bins.
    B : (N_bin, N_sn)
        Binning matrix from bin_sn_data.
    z_raw : (N_sn,)
        Raw SN redshifts.

    Returns
    -------
    z_eff : (N_bin,)
        Effective redshifts. For bins with 1 SN, z_eff == z_bin.
    """
    from scipy.optimize import brentq
    from .cosmology import compute_distance_modulus

    mu_ref_raw = compute_distance_modulus(ref_cosmo, z_raw)
    mu_ref_binned = B @ mu_ref_raw  # target: mu_ref(z_eff) should equal this

    n_bins = len(z_bin)
    z_eff = np.zeros(n_bins)

    for j in range(n_bins):
        target = mu_ref_binned[j]
        # If bin has 0 or 1 SN, no correction needed
        n_sn_in_bin = np.sum(B[j] > 0)
        if n_sn_in_bin <= 1:
            z_eff[j] = z_bin[j]
            continue

        def f(z):
            return compute_distance_modulus(ref_cosmo, np.array([z]))[0] - target

        z_lo = max(1e-5, z_bin[j] - 0.1)
        z_hi = z_bin[j] + 0.1
        # Widen bracket if needed
        if f(z_lo) * f(z_hi) > 0:
            z_lo = max(1e-5, z_bin[j] - 0.5)
            z_hi = z_bin[j] + 0.5

        try:
            z_eff[j] = brentq(f, z_lo, z_hi, xtol=1e-10)
        except ValueError:
            z_eff[j] = z_bin[j]

    return z_eff


if __name__ == "__main__":
    # Test loading
    try:
        bao = load_official_bao_data()
        print_bao_summary(bao)

        print("\n" + "=" * 70)
        print("WHITENED DATA")
        print("=" * 70)
        print("\nWhitened data vector:")
        for i, (obs, w) in enumerate(zip(bao['observables'], bao['whitened_data'])):
            print(f"  {i+1}: {w:+.4f} ({obs['quantity']} at z={obs['z']:.3f})")

        # Verify whitening
        w = bao['whitened_data']
        reconstructed_cov = np.outer(w, w)
        print(f"\nWhitened data has unit covariance: "
              f"||I - W W^T||_F = {np.linalg.norm(np.eye(len(w)) - np.cov(w.reshape(1,-1))):.2e}")

    except FileNotFoundError as e:
        print(f"Data not found: {e}")
