import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.stats import poisson
import pandas as pd
import matplotlib as mpl
import matplotlib.gridspec 
from time import perf_counter
from jax.flatten_util import ravel_pytree


# ============================================================
# Streamlit Configuration
# ============================================================
st.set_page_config(layout="wide", page_title="Neutron Scattering Experiment Simulator")

# ============================================================
# Custom CSS for better styling
# ============================================================
st.markdown("""
<style>
:root {
    --bg-main: var(--background-color);
    --bg-secondary: var(--secondary-background-color);
    --text-color: var(--text-color);
    --accent: var(--primary-color);
}

.main-header {
    font-size: 2.5rem;
    color: var(--accent);
    text-align: center;
    margin-bottom: 1rem;
}

.info-box,
.score-box,
.time-box {
    background-color: var(--bg-secondary);
    color: var(--text-color);
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid var(--accent);
    margin-bottom: 1rem;
}

.progress-container {
    background-color: #2A2F3A;
    border-radius: 10px;
    height: 20px;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    border-radius: 10px;
    transition: width 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Initialize Session State
# ============================================================
def initialize_session_state():
    if 'game_over' not in st.session_state:
        st.session_state.game_over = False
    if 'used_time' not in st.session_state:
        st.session_state.used_time = 0.0
    if 'all_scans' not in st.session_state:
        st.session_state.all_scans = get_empty_scans_df()
    if 'show_theory' not in st.session_state:
        st.session_state.show_theory = True
    if 'temp_slider' not in st.session_state:
        st.session_state.temp_slider = 50.0
    if 'ct_slider' not in st.session_state:
        st.session_state.ct_slider = 61.0
    if 'np_slider' not in st.session_state:
        st.session_state.np_slider = 31
    if 'er_slider' not in st.session_state:
        st.session_state.er_slider = 6.0
    if 'sc_slider' not in st.session_state:
        st.session_state.sc_slider = 5.0
    if 'profiling_output' not in st.session_state:
        st.session_state.profiling_output = ""
    if 'enable_profiling' not in st.session_state:
        st.session_state.enable_profiling = False
    if 'jax_warmed_up' not in st.session_state:
        st.session_state.jax_warmed_up = False

# ============================================================
# Physics Constants
# ============================================================
kB = 0.08617  # Boltzmann constant in meV/K
min_t, max_t = 2, 270

# ============================================================
# Game Settings
# ============================================================
TIME_BUDGET = 12 * 3600  # 12 hours in seconds

def get_empty_scans_df():
    """Return an empty DataFrame with the correct schema"""
    return pd.DataFrame({
        'T': pd.Series(dtype='float32'),
        'Energy': pd.Series(dtype='float32'),
        'counts_per_sec': pd.Series(dtype='float32'),
        'error_l': pd.Series(dtype='float32'),
        'error_h': pd.Series(dtype='float32'),
        'lam': pd.Series(dtype='float32'),
        'amp': pd.Series(dtype='float32'),
        'E0': pd.Series(dtype='float32'),
        'hwhm': pd.Series(dtype='float32'),
        'bg': pd.Series(dtype='float32'),
        'counts': pd.Series(dtype='int64'),
        'count_time': pd.Series(dtype='float32'),
        'pressurized': pd.Series(dtype='bool'),
        'FI_hwhm': pd.Series(dtype='float32'),
    })


# ============================================================
# Damped Harmonic Oscillator Model
# ============================================================

# -----------------------
# Physical constants
# -----------------------
kB = 0.086173324  # meV / K


# -----------------------
# Bose factor
# -----------------------
def bose(E, T):
    absE = jnp.abs(E)
    n = 1.0 / (jnp.exp(absE / (kB * T)) - 1.0)
    return jnp.where(E >= 0, n + 1.0, n)


# -----------------------
# Smooth utilities
# -----------------------
def smooth_step(x, x0, w):
    return 0.5 * (1.0 + jnp.tanh((x - x0) / w))


# -----------------------
# Temperature-dependent HWHM
# -----------------------
def temperature_dependent_hwhm(T, hwhm_params):
    """
    hwhm_params = {
        "hwhm0": float,
        "slopes": jnp.array([s1, s2, s3]),
        "T1": float,
        "T2": float,
        "Tmin": float,
        "width": float,
    }
    """
    h0 = hwhm_params["hwhm0"]
    s1, s2, s3 = hwhm_params["slopes"]

    T1 = hwhm_params["T1"]
    T2 = hwhm_params["T2"]
    Tmin = hwhm_params["Tmin"]
    w = hwhm_params["width"]

    h1 = h0 * (1.0 + s1 * (T - Tmin))
    h1_T1 = h0 * (1.0 + s1 * (T1 - Tmin))

    h2 = h1_T1 + h0 * s2 * (T - T1)
    h2_T2 = h1_T1 + h0 * s2 * (T2 - T1)

    h3 = h2_T2 + h0 * s3 * (T - T2)

    s_1 = smooth_step(T, T1, w)
    s_2 = smooth_step(T, T2, w)

    return (1.0 - s_1) * h1 + s_1 * (1.0 - s_2) * h2 + s_2 * h3


# -----------------------
# Damped Harmonic Oscillator
# -----------------------
def damped_harmonic_oscillator(E, T, dho_params):
    """
    dho_params = { "amp":, "E0":, "hwhm": }
    """
    amp = dho_params["amp"]
    E0 = dho_params["E0"]
    hwhm = dho_params["hwhm"]

    lorentz = (
        hwhm / ((E - E0) ** 2 + hwhm ** 2)
        - hwhm / ((E + E0) ** 2 + hwhm ** 2)
    )

    prefactor = amp / (E0 * jnp.pi)
    return jnp.abs(prefactor * lorentz) * bose(E, T)

def params_at_temp(T, params, pressurized):
    """
    Unified parameter tree:

    params = {
        "amp0": float,
        "E00": float,
        "bg0": float,
        "hwhm0": float,
        "slopes": {
            "pressurized": jnp.array([s1, s2, s3]),
            "non_pressurized": jnp.array([s1, s2, s3]),
        },
        "T1": float,
        "T2": float,
        "Tmin": float,
        "width": float,
    }
    """
    amp = params["amp0"] * (1.0 - 0.3 * (T - 2.0) / 268.0)
    bg = params["bg0"] * (1.0 + 2.5 * (T - 2.0) / 268.0)
    E0 = params["E00"] * (1.0 - 0.1 * (T - 2.0) / 268.0)

    slopes = jnp.where(
        pressurized,
        params["slopes"]["pressurized"],
        params["slopes"]["non_pressurized"],
    )

    hwhm = temperature_dependent_hwhm(
        T,
        {
            "hwhm0": params["hwhm0"],
            "slopes": slopes,
            "T1": params["T1"],
            "T2": params["T2"],
            "Tmin": params["Tmin"],
            "width": params["width"],
        },
    )
    return {'amp':amp, 'bg':bg, 'E0':E0, 'hwhm':hwhm}


def model(E, T, params, pressurized):
    """
    Unified parameter tree:

    params = {
        "amp0": float,
        "E00": float,
        "bg0": float,
        "hwhm0": float,
        "slopes": {
            "pressurized": jnp.array([s1, s2, s3]),
            "non_pressurized": jnp.array([s1, s2, s3]),
        },
        "T1": float,
        "T2": float,
        "Tmin": float,
        "width": float,
    }
    """
    p = params_at_temp(T, params, pressurized)
    amp, bg, E0, hwhm = p['amp'], p['bg'], p['E0'], p['hwhm']

    signal = damped_harmonic_oscillator(
        E,
        T,
        {"amp": amp, "E0": E0, "hwhm": hwhm},
    )

    return signal + bg


# -----------------------
# Vectorized rate model
# -----------------------
def lambda_model(params, E, T, pressurized):
    return jax.vmap(
        model,
        in_axes=(0, 0, None, 0)
    )(E, T, params, pressurized)

def hwhm_fisher_per_point(E, T, params, pressurized, count_time):
    """
    Per-point Fisher information for HWHM only.
    
    Returns:
        FI_i : jnp.ndarray, shape (N_points,)
    """
    # Temperature-dependent parameters (fixed except hwhm)
    p_T = params_at_temp(T, params, pressurized)

    amp = p_T["amp"]
    E0 = p_T["E0"]
    bg = p_T["bg"]
    hwhm0 = p_T["hwhm"]

    # Rate at each point
    def lambda_i(hwhm):
        return (
            damped_harmonic_oscillator(
                E,
                T,
                {"amp": amp, "E0": E0, "hwhm": hwhm},
            )
            + bg
        )

    lam = lambda_i(hwhm0)
    lam = jnp.clip(lam, 1e-12, jnp.inf)

    # Pointwise derivative dλ_i / dΓ
    dlam_dhwhm = jax.jacobian(lambda h: lambda_i(h))(hwhm0)

    # Per-point Fisher information
    FI_i = count_time * (dlam_dhwhm ** 2) / lam

    return FI_i

@st.cache_resource(show_spinner="Compiling JAX kernels…")
def warmup_jax():
    """
    Compile and cache all JAX-jitted functions.
    Returned functions overwrite Python versions.
    """

    # -----------------------
    # JIT wrappers
    # -----------------------
    jitted_model = jax.jit(model)
    jitted_lambda_model = jax.jit(lambda_model)
    jitted_hwhm_fisher = jax.jit(hwhm_fisher_per_point)
    jitted_fisher_information = jax.jit(fisher_information)

    # -----------------------
    # Dummy inputs for compilation
    # -----------------------
    E = jnp.linspace(-5.0, 5.0, 32)
    T = jnp.full_like(E, 50.0)
    press = jnp.zeros_like(E, dtype=bool)
    ct = 10.0

    # Force compilation
    _ = jitted_model(E, 50.0, TRUE_PARAMS, False)
    _ = jitted_lambda_model(TRUE_PARAMS, E, T, press)
    _ = jitted_hwhm_fisher(E, 50.0, TRUE_PARAMS, False, ct)
    _ = jitted_fisher_information(TRUE_PARAMS, E, T, press, ct)

    return {
        "model": jitted_model,
        "lambda_model": jitted_lambda_model,
        "hwhm_fisher_per_point": jitted_hwhm_fisher,
        "fisher_information": jitted_fisher_information,
    }

# -----------------------
# Poisson scan sampling
# -----------------------
def sample_poisson_scan(
    T,
    params,
    npts,
    E_range,
    count_time,
    scan_center,
    pressurized=False,
):
    Es = np.linspace(
        scan_center - E_range / 2,
        scan_center + E_range / 2,
        npts,
    )
    Es_jax = jnp.array(Es)
    T_arr = jnp.full_like(Es_jax, T)
    pressurized_arr = jnp.ones(len(Es), dtype=bool)*pressurized
    
    lam = np.array(lambda_model(params, Es_jax, T_arr, pressurized_arr))
    lam = np.clip(lam, 1e-12, np.inf)
    T_params = params_at_temp(T, params, pressurized)

    counts = np.random.poisson(lam * count_time)
    rate = counts / count_time

    alpha = 0.32
    lower = poisson.ppf(alpha / 2, counts) / count_time
    upper = poisson.ppf(1 - alpha / 2, counts) / count_time

    errors = np.vstack([rate - lower, upper - rate])

    FI_hwhm = np.array(
        hwhm_fisher_per_point(
            Es_jax,
            T,
            params,
            pressurized,
            count_time,
        )
    )

    return Es, rate, errors, lam, counts, T_params, FI_hwhm


# -----------------------
# Fisher Information Matrix
# -----------------------
def fisher_information(params, E, T, pressurized, count_time):
    lam = lambda_model(params, E, T, pressurized)
    lam = jnp.clip(lam, 1e-12, jnp.inf)

    theta0 = pack_inference_params(params)

    def rate_fn(theta):
        p = unpack_inference_params(theta, params)
        return lambda_model(p, E, T, pressurized)

    J = jax.jacobian(rate_fn)(theta0)  # (N_data, N_params)

    W = count_time / lam
    FI = J.T @ (J * W[:, None])

    return FI

def pack_inference_params(params):
    """
    Parameters we want uncertainties for:
    amp0, E00, bg0, hwhm0,
    pressurized slopes (3),
    non-pressurized slopes (3)
    """
    return jnp.concatenate([
        jnp.array([params["amp0"],
                   params["E00"],
                   params["bg0"],
                   params["hwhm0"]]),
        params["slopes"]["pressurized"],
        params["slopes"]["non_pressurized"],
    ])
def unpack_inference_params(theta, template):
    p = dict(template)

    p["amp0"]  = theta[0]
    p["E00"]   = theta[1]
    p["bg0"]   = theta[2]
    p["hwhm0"] = theta[3]

    p["slopes"] = {
        "pressurized":     theta[4:7],
        "non_pressurized": theta[7:10],
    }

    return p

def compute_total_fisher_matrix_all_params():
    """
    Compute the full Fisher Information Matrix for all model parameters
    using all scans currently in session_state.
    """

    if len(st.session_state.all_scans) == 0:
        return None, None

    # Parameter ordering (MUST match plotting)
    param_names = [
        "amp_2K",
        "E0_2K",
        "bg_2K",
        "hwhm_2K",
        "P_slope_1",
        "P_slope_2",
        "P_slope_3",
        "NP_slope_1",
        "NP_slope_2",
        "NP_slope_3",
    ]
    df = st.session_state.all_scans
    E = jnp.asarray(df["Energy"].values)
    T = jnp.asarray(df["T"].values)
    pressurized = jnp.asarray(df["pressurized"].values)
    count_time = jnp.asarray(df["count_time"].values)
    # params_t = params_at_temp(T, TRUE_PARAMS, pressurized)
    FI_total = fisher_information(TRUE_PARAMS, E, T, pressurized, count_time)

    return FI_total, param_names

# Add this function after compute_total_fisher_matrix_all_params()
def compute_fisher_log_det(FI_matrix):
    """Compute log determinant of Fisher Information Matrix"""
    if FI_matrix is None:
        return -np.inf
    
    try:
        # Add small regularization to ensure positive definiteness
        reg_matrix = FI_matrix + 1e-10 * np.eye(FI_matrix.shape[0])
        sign, logdet = np.linalg.slogdet(reg_matrix)
        return logdet
    except np.linalg.LinAlgError:
        return -np.inf
# -----------------------
# Example TRUE parameter tree
# -----------------------
TRUE_PARAMS = {
    "amp0": 2.0,
    "E00": 5.0,
    "bg0": 1.5,
    "hwhm0": 0.15,
    "slopes": {
        "non_pressurized": jnp.array([0.1 / 48.0, 0.8 / 100.0, 0.3 / 120.0]),
        "pressurized": jnp.array([0.2 / 48.0, 1.2 / 100.0, 0.5 / 120.0]),
    },
    "T1": 50.0,
    "T2": 150.0,
    "Tmin": 2.0,
    "width": 8.0,
}

def format_time_delta(seconds):
    """Format seconds into flexible hh:mm:ss format"""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_remaining = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds_remaining:02d}"

def add_scan(pressurized=False):
    if st.session_state.game_over:
        return
    
    T = st.session_state.temp_slider
    count_time = st.session_state.ct_slider
    npts = st.session_state.np_slider
    E_range = st.session_state.er_slider
    scan_center = st.session_state.sc_slider
    
    total_scan_time = count_time * npts
    
    # Check if we have enough time
    end_with_game_over = False
    if st.session_state.used_time + total_scan_time > TIME_BUDGET:
        # Calculate how many points we can actually measure
        remaining_time = TIME_BUDGET - st.session_state.used_time
        if remaining_time <= 0:
            trigger_game_over()
            return
            
        npts_possible = int(remaining_time / count_time)
        end_with_game_over = True

        # Use the smaller number of points
        npts = npts_possible
        total_scan_time = count_time * npts

    st.session_state.used_time += total_scan_time

    # Sample actual data
    Es, counts_per_sec, errors, lam, counts, params, FI_hwhm = sample_poisson_scan(
        T=T,
        params=TRUE_PARAMS,
        npts=npts,
        E_range=E_range,
        count_time=count_time,
        scan_center=scan_center,
        pressurized=pressurized,
    )


    # --- Construct DataFrame (identical columns as before) ---
    new_df = pd.DataFrame({
        "T": T,
        "Energy": Es,
        "counts_per_sec": counts_per_sec,
        "error_l": errors[0],
        "error_h": errors[1],
        "lam": lam,
        "counts": counts,
        "count_time": count_time,
        "pressurized": pressurized,
        'hwhm': params['hwhm'],
        'FI_hwhm': FI_hwhm,
    })


    st.session_state.all_scans = (
        new_df
        if len(st.session_state.all_scans)==0
        else pd.concat([st.session_state.all_scans, new_df], ignore_index=True)
    )

    if end_with_game_over:
        trigger_game_over()

def clear_plots():
    """Clear all data and reset the game"""
    st.session_state.all_scans = get_empty_scans_df()
    
    st.session_state.used_time = 0.0
    st.session_state.game_over = False
    st.rerun()

def trigger_game_over():
    """Trigger game over state"""
    st.session_state.game_over = True

def create_custom_progress_bar(progress_ratio):
    """Create a custom progress bar that handles values > 1.0"""
    # Clamp progress ratio between 0 and 1 for display
    display_ratio = min(1.0, progress_ratio)
    
    # Determine color based on progress
    if progress_ratio > 1.0:
        color = "#DC2626"  # Red for exceeded
    elif display_ratio > 0.9:
        color = "#F59E0B"  # Orange for high
    elif display_ratio > 0.7:
        color = "#3B82F6"  # Blue for medium
    else:
        color = "#10B981"  # Green for low
    
    # Create custom HTML progress bar
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {display_ratio * 100}%; background-color: {color};">
        </div>
    </div>
    """
    return progress_html, display_ratio

def create_plots():
    """Create and return matplotlib figures"""
    # Set style    
    plt.style.use('dark_background')

    mpl.rcParams.update({
        'figure.facecolor': '#0E1117',
        'axes.facecolor': '#161B22',
        'axes.edgecolor': '#E5E7EB',
        'axes.labelcolor': '#E5E7EB',
        'text.color': '#E5E7EB',
        'xtick.color': '#E5E7EB',
        'ytick.color': '#E5E7EB',
        'grid.color': '#374151',
    })

    fig = plt.figure(figsize=(10, 6))
    gs2 = matplotlib.gridspec.GridSpec(2, 2, height_ratios=[1, 1], figure=fig)
    ax1 = plt.subplot(gs2[0, 0])
    ax2 = plt.subplot(gs2[0, 1])
    # gs2.update(hspace=0.3, wspace=0.3)
    ax3 = plt.subplot(gs2[1, :])
    fig = plt.gcf()

    fig.suptitle('Neutron Scattering Experiment Simulator', fontsize=16, fontweight='bold')
    
    # Colormaps
    cmap_non_pressurized = plt.get_cmap('Wistia')
    cmap_pressurized = plt.get_cmap('Wistia')
    
    # Plot non-pressurized data
    ax1.set_title("Non-Pressurized Sample", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Energy [meV]", fontsize=9)
    ax1.set_ylabel("Counts per Second", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    non_press_scans = st.session_state.all_scans[~st.session_state.all_scans['pressurized']]
    if len(non_press_scans) > 0:
        for T, df in non_press_scans.groupby('T'):
            color_val = (T - min_t) / (max_t - min_t)
            color = cmap_non_pressurized(color_val)
            ind = np.argsort(df['Energy'].values)
            label = f'T={T:.1f}K ({len(df)} pts)'
            ax1.errorbar(df['Energy'].values[ind], df['counts_per_sec'].values[ind],
                        yerr=(df['error_l'], df['error_h']),
                        fmt='o', color=color, alpha=0.7, markersize=3,
                        label=label, capsize=2, capthick=1, elinewidth=1)
            
            if st.session_state.show_theory:
                ax1.plot(df['Energy'].values[ind], df['lam'].values[ind],
                        '--', color=color, linewidth=1.5, alpha=0.8)
        
        if len(non_press_scans.groupby('T')) > 0:
            ax1.legend(loc='upper right', framealpha=0.9, fontsize=8)
    
    # Plot pressurized data
    ax2.set_title("Pressurized Sample", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Energy [meV]", fontsize=9)
    ax2.set_ylabel("Counts per Second", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    press_scans = st.session_state.all_scans[st.session_state.all_scans['pressurized']]
    if len(press_scans) > 0:
        for T, df in press_scans.groupby('T'):
            color_val = (T - min_t) / (max_t - min_t)
            color = cmap_pressurized(color_val)
            ind = np.argsort(df['Energy'].values)
            label = f'T={T:.1f}K ({len(df)} pts)'
            ax2.errorbar(df['Energy'].values[ind], df['counts_per_sec'].values[ind],
                        yerr=(df['error_l'], df['error_h']),
                        fmt='o', color=color, alpha=0.7, markersize=3,
                        label=label, capsize=2, capthick=1, elinewidth=1)
            
            if st.session_state.show_theory:
                ax2.plot(df['Energy'].values[ind], df['lam'].values[ind],
                        '--', color=color, linewidth=1.5, alpha=0.8)
        
        if len(press_scans.groupby('T')) > 0:
            ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
    
    # Plot HWHM vs Temperature
    ax3.set_title("Temperature-Dependent Linewidth HWHM(T)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Temperature [K]", fontsize=9)
    ax3.set_ylabel("HWHM [meV]", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.set_xlim(min_t-2, max_t + 1)
    
    if len(st.session_state.all_scans) > 0:
        # Non-pressurized measurements
        non_press_scans = st.session_state.all_scans[~st.session_state.all_scans['pressurized']]
        for T, df in non_press_scans.groupby('T'):
            if len(df) > 0:
                hwhm_err = 1 / jnp.sqrt(jnp.clip(df['FI_hwhm'].values.sum(), 1e-3)) # compute_hwhm_fisher(df)
                ax3.errorbar(T, df['hwhm'].mean(), hwhm_err,
                            linestyle='none', color='gold', linewidth=2,
                            marker='o', markersize=6, markeredgecolor='gold',
                            markeredgewidth=0.5, capsize=4, capthick=2,
                            label='Non-Pressurized' if T == non_press_scans['T'].min() else None)
        
        # Pressurized measurements
        press_scans = st.session_state.all_scans[st.session_state.all_scans['pressurized']]
        for T, df in press_scans.groupby('T'):
            if len(df) > 0:
                hwhm_err = 1 / jnp.sqrt(jnp.clip(df['FI_hwhm'].values.sum(), 1e-3)) # compute_hwhm_fisher(df)
                ax3.errorbar(T, df['hwhm'].mean(), hwhm_err,
                            linestyle='none', color='red', linewidth=2,
                            marker='o', markersize=6, markeredgecolor='red',
                            markeredgewidth=0.5, capsize=4, capthick=2,
                            label='Pressurized' if T == press_scans['T'].min() else None)
        
    # Theoretical guides
    if st.session_state.show_theory:
        T_guide = np.linspace(2, 270, 100)
        hwhm_guide_non = [
            temperature_dependent_hwhm(
                T_val,
                {
                    "hwhm0": TRUE_PARAMS["hwhm0"],
                    "slopes": TRUE_PARAMS["slopes"]["non_pressurized"],
                    "T1": TRUE_PARAMS["T1"],
                    "T2": TRUE_PARAMS["T2"],
                    "Tmin": TRUE_PARAMS["Tmin"],
                    "width": TRUE_PARAMS["width"],
                }
            )
            for T_val in T_guide
        ]
        hwhm_guide_press = [
            temperature_dependent_hwhm(
                T_val,
                {
                    "hwhm0": TRUE_PARAMS["hwhm0"],
                    "slopes": TRUE_PARAMS["slopes"]["pressurized"],
                    "T1": TRUE_PARAMS["T1"],
                    "T2": TRUE_PARAMS["T2"],
                    "Tmin": TRUE_PARAMS["Tmin"],
                    "width": TRUE_PARAMS["width"],
                }
            )
            for T_val in T_guide
        ]

        # hwhm_guide_non = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, False) for T_val in T_guide]
        # hwhm_guide_press = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, True) for T_val in T_guide]
        

        ax3.plot(T_guide, hwhm_guide_non, '--', color='gold', alpha=0.5, label='Non-Pressurized (expected)')
        ax3.plot(T_guide, hwhm_guide_press, '--', color='red', alpha=0.5, label='Pressurized (expected)')
    
        # Legend
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax3.legend(by_label.values(), by_label.keys(), fontsize=8)
    
    plt.tight_layout()
    return fig

# ============================================================
# Main Streamlit App
# ============================================================
def main():
    # Initialize session state
    initialize_session_state()
    
    # JAX warmup on first run
    if not st.session_state.jax_warmed_up:
        compiled = warmup_jax()

        globals()["model"] = compiled["model"]
        globals()["lambda_model"] = compiled["lambda_model"]
        globals()["hwhm_fisher_per_point"] = compiled["hwhm_fisher_per_point"]
        globals()["fisher_information"] = compiled["fisher_information"]

        st.session_state.jax_warmed_up = True
        
    st.markdown('<h1 class="main-header">Neutron Scattering Experiment Simulator</h1>', unsafe_allow_html=True)
    
    # Header info
    st.markdown("""
Imagine you are an experimentalist that wants to understand how pressure effects your crystal. You have two samples of your crystal, and you have applied a 
uniaxial pressure to one sample. You are making scans at different energy transfer values (the x axis) and observing the neutron intensity (the y axis).
You believe that the bump you observe has a width that changes with temperature and changes differently for the two samples. Your goal is to use your limited 
beamtime to measure as much useful information as possible. You have 12 hours and the controls on the left. You are scored based on the precision of the 
parameters governing the width of the peak as a function of temperature. The game will stop when you request more time than is available. Good luck!
                
""")
    # Sidebar for controls
    st.markdown("---")
    with st.sidebar:
        st.markdown("# Experiment Controls")
        total_scan_time = st.session_state.ct_slider * st.session_state.np_slider
        remaining_time = TIME_BUDGET - st.session_state.used_time

        if st.button("Scan Non-Pressurized Sample", 
                    disabled=st.session_state.game_over or remaining_time <= 0,
                    help="Run a scan at non-pressurized conditions",
                    type="secondary"):
            add_scan(pressurized=False)
        
        if st.button("Scan Pressurized Sample", 
                    disabled=st.session_state.game_over or remaining_time <= 0,
                    help="Run a scan at pressurized conditions",
                    type="secondary"):
            add_scan(pressurized=True)

        if st.session_state.game_over:
            st.error("⏰ Game Over! Time budget exhausted!")
        if st.button("Restart Game"):
            clear_plots()
        
        if st.button("Theory Lines On/Off", 
                help="Show/hide theoretical prediction lines"):
            st.session_state.show_theory = not st.session_state.show_theory
            st.rerun()

        st.markdown("---")
        st.markdown("### Scan Parameters")
        
        # Sliders
        st.slider(
            "Temperature [K]",
            min_value=2.0,
            max_value=270.0,
            key='temp_slider',
            step=2.0,
            help="Center temperature for the scan"
        ) 
        st.slider(
            "Counting Time per Point [s]",
            min_value=1.0,
            max_value=800.0,
            step=30.0,
            key="ct_slider",
            help="Time spent measuring each energy point"
        )
        
        st.slider(
            "Number of Points",
            min_value=1,
            max_value=100,
            key='np_slider',
            step=10,
            help="Number of energy points in the scan"
        )
        
        st.slider(
            "Energy Range [meV]",
            min_value=1.0,
            max_value=8.0,
            key='er_slider',
            step=0.5,
            help="Total energy range centered on scan center"
        )
        
        st.slider(
            "Energy Center [meV]",
            min_value=0.1,
            max_value=20.0,
            key='sc_slider',
            step=0.5,
            help="Center energy for the scan"
        )
        
        # Calculate total scan time
        
        st.info(f"**This scan will take:** {format_time_delta(total_scan_time)}")
        
        if total_scan_time > remaining_time and not st.session_state.game_over:
            st.warning(f"⚠️ Not enough time for full scan! You only have {format_time_delta(remaining_time)} remaining.")
        
        st.markdown("---")

    # Main content area
    # Create and display plots
    fig = create_plots()
    st.pyplot(fig)
    
    # Data summary

    if len(st.session_state.all_scans) > 0:


        # High Score Section
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        # Calculate Fisher log determinant
        FI_full, param_names_full = compute_total_fisher_matrix_all_params()
        current_score = compute_fisher_log_det(FI_full)

        # Initialize high score in session state if not present
        if 'high_score' not in st.session_state:
            st.session_state.high_score = -np.inf
        if 'best_run_data' not in st.session_state:
            st.session_state.best_run_data = None

        # Update high score if current score is better
        if current_score > st.session_state.high_score and not np.isinf(current_score):
            st.session_state.high_score = current_score
            st.session_state.best_run_data = {
                'score': current_score,
                'scans': st.session_state.all_scans.copy(),
                'time_used': st.session_state.used_time,
                'n_measurements': len(st.session_state.all_scans)
            }

        # Display metrics
        with col1:
            st.metric(
                "Current Score",
                f"{current_score:.2f}" if not np.isinf(current_score) else "No data",
                delta=None
            )

        with col2:
            st.metric(
                "🏆 High Score",
                f"{st.session_state.high_score:.2f}" if st.session_state.high_score > -np.inf else "No data",
                delta=f"{current_score - st.session_state.high_score:.2f}" if st.session_state.high_score > -np.inf and not np.isinf(current_score) else None
            )


        # Data summary and other expanders continue here...
        st.markdown("---")

        FI_full, param_names_full = compute_total_fisher_matrix_all_params()

        with st.expander("Experimental Data Review", expanded=True):

            summary_data = {
                "Total Measurements": len(st.session_state.all_scans),
                "Non-Pressurized Scans": len(
                    st.session_state.all_scans[
                        ~st.session_state.all_scans["pressurized"]
                    ]
                ),
                "Pressurized Scans": len(
                    st.session_state.all_scans[
                        st.session_state.all_scans["pressurized"]
                    ]
                ),
                "Unique Temperatures": st.session_state.all_scans["T"].nunique(),
                "Temperature Coverage": (
                    f"{st.session_state.all_scans['T'].min():.1f} K → "
                    f"{st.session_state.all_scans['T'].max():.1f} K"
                ),
                "Total Measurement Time": f"{st.session_state.used_time/3600:.2f} h",
                "Remaining Time": f"{max(0, (TIME_BUDGET - st.session_state.used_time)/3600):.2f} h",
                "Time Budget Used": f"{min(100, (st.session_state.used_time/TIME_BUDGET)*100):.1f} %",
            }

            cols = st.columns(3)
            for i, (label, value) in enumerate(summary_data.items()):
                cols[i % 3].metric(label, value)

        with st.expander("Fisher Information Matrix", expanded=True):

            if FI_full is not None:
                fig, ax = plt.subplots(figsize=(8, 6))

                short_names = [
                    "Amp(2K)", "E₀(2K)", "BG(2K)", "HWHM(2K)",
                    "P Slope 1", "P Slope 2", "P Slope 3",
                    "NP Slope 1", "NP Slope 2", "NP Slope 3",
                ]

                im = ax.imshow(
                    FI_full,
                    cmap="RdBu_r",
                    # vmin=-1,
                    # vmax=1,
                    aspect="auto",
                )

                ax.set_xticks(range(len(short_names)))
                ax.set_yticks(range(len(short_names)))
                ax.set_xticklabels(short_names, rotation=45, ha="right")
                ax.set_yticklabels(short_names)

                ax.set_title("Fisher Information Matrix")

                plt.colorbar(im, ax=ax, label="Fᵢⱼ")

                threshold = 0.1 * np.max(np.abs(FI_full))
                for i in range(FI_full.shape[0]):
                    for j in range(FI_full.shape[1]):
                        if abs(FI_full[i, j]) > threshold:
                            ax.text(
                                j, i,
                                f"{FI_full[i, j]:.1e}",
                                ha="center",
                                va="center",
                                fontsize=7,
                                color="white" if FI_full[i, j] > 0.5 else "black",
                            )

                plt.tight_layout()
                st.pyplot(fig)

                with st.expander("How to read this matrix"):
                    st.markdown("""
        **What this shows**

        - **Diagonal elements**  
        → How strongly each parameter is constrained by the experiment.

        - **Off-diagonal elements**  
        → Parameter correlations (degeneracies).

        - **Bright blocks**  
        → Groups of parameters informed by the same physics.

        **What you want to see**

        - Strong diagonals  
        - Weak off-diagonals  
        - No large, coherent correlation blocks between unrelated parameters
        """)
        with st.expander("Covariance Matrix", expanded=True):

            try:
                covariance = np.linalg.pinv(FI_full + 1e-10 * np.eye(FI_full.shape[0]))

                fig, ax = plt.subplots(figsize=(8, 6))

                im = ax.imshow(
                    covariance,
                    cmap="RdYlBu_r",
                    aspect="auto",
                )

                ax.set_xticks(range(len(short_names)))
                ax.set_yticks(range(len(short_names)))
                ax.set_xticklabels(short_names, rotation=45, ha="right")
                ax.set_yticklabels(short_names)

                ax.set_title("Covariance Matrix")

                plt.colorbar(im, ax=ax, label="Covariance")

                for i in range(covariance.shape[0]):
                    for j in range(i + 1):
                        ax.text(
                            j, i,
                            f"{covariance[i, j]:.1e}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )

                plt.tight_layout()
                st.pyplot(fig)

            except np.linalg.LinAlgError:
                st.warning("Covariance matrix could not be computed (singular Fisher matrix).")
                       
    st.markdown('---')

    with st.expander("### The Physical Model"):
        st.markdown("""
        The physical model in this case is intended as a surrogate model and therefore is not 
        exactly the same as what one would expect. But the idea is to be as close as is reasonable.
                    
        The intensity at an energy E and time T is a sum of the model and temperature dependent backgroudn
                            """)
        st.latex(r"""
        I(E, T) = S(E, T) + B(T)
        """)

        st.markdown("The scattering function S is modeled as a damped harmonic oscillator:")
        st.latex(r"""
        S(E, T)
        =
        \left|
        \frac{A(T)}{\pi E_0(T)}
        \left[
        \frac{\Gamma(T)}{(E - E_0(T))^2 + \Gamma(T)^2}
        -
        \frac{\Gamma(T)}{(E + E_0(T))^2 + \Gamma(T)^2}
        \right]
        \right|
        \cdot n_B(E,T)
        """)

        st.markdown("The Bose population factor is")
        st.latex(r"""
        n_B(E,T) =
        \begin{cases}
        \displaystyle \frac{1}{e^{|E|/(k_B T)} - 1} + 1, & E \ge 0 \\[6pt]
        \displaystyle \frac{1}{e^{|E|/(k_B T)} - 1}, & E < 0
        \end{cases}
        """)
        st.latex(r"""
        k_B = 0.08617\ \text{meV/K}
        """)

        st.markdown("#### Temperature Dependence of Parameters")
        st.markdown('A is the amplitude, B is the background, and E0 is the center of the DHO.')

        st.latex(r"""
        A(T) = A_0 \left( 1 - 0.3 \frac{T - 2}{268} \right)
        """)

        st.latex(r"""
        B(T) = B_0 \left( 1 + 2.5 \frac{T - 2}{268} \right)
        """)

        st.latex(r"""
        E_0(T) = E_{00} \left( 1 - 0.1 \frac{T - 2}{268} \right)
        """)

        st.markdown("#### Temperature-Dependent Linewidth (HWHM)")

        st.latex(r"""
        \Gamma(T)
        =
        w_1(T)\,\Gamma_1(T)
        +
        w_2(T)\,\Gamma_2(T)
        +
        w_3(T)\,\Gamma_3(T)
        """)

        st.markdown("The width is modeled as (smooth) piecewise linear segments:")
        st.latex(r"""
        \Gamma_1(T) = \Gamma_0 \left[ 1 + m_1 (T - T_{\min}) \right]
        """)
        st.latex(r"""
        \Gamma_2(T) = \Gamma_1(T_1) + \Gamma_0 m_2 (T - T_1)
        """)
        st.latex(r"""
        \Gamma_3(T) = \Gamma_2(T_2) + \Gamma_0 m_3 (T - T_2)
        """)

        st.markdown("The transitions use a tanh function for smoothing")
        st.latex(r"""
        s_i(T) = \frac{1}{2}
        \left[
        1 + \tanh\!\left( \frac{T - T_i}{\Delta T} \right)
        \right]
        """)
        st.latex(r"""
        w_1 = 1 - s_1, \quad
        w_2 = s_1 (1 - s_2), \quad
        w_3 = s_2
        """)

        st.markdown("The slopes depend on pressure condition:")
        st.latex(r"""
        \{m_1, m_2, m_3\}
        =
        \begin{cases}
        \{m_1^{(p)}, m_2^{(p)}, m_3^{(p)}\}, & \text{pressurized} \\
        \{m_1^{(np)}, m_2^{(np)}, m_3^{(np)}\}, & \text{non-pressurized}
        \end{cases}
        """)

        st.markdown("#### Measurement Process")
        st.markdown('Measurements are Poisson observations that depend linearly on count time.')
        st.latex(r"""
        N_i \sim \mathrm{Poisson}\!\left( I(E_i, T) \cdot t_i \right)
        """)

    with st.expander('### Measurement Uncertainty'):
        st.markdown("Let a measurement $y$ depend on location $x$ and parameters $\\theta$:")

        st.latex(r"y_i \sim \text{Poisson}(f(x_i; \theta))")

        st.markdown(r'For the Poisson distribution, the probability of observing a count n given a rate $\lambda$ is:')
        st.latex(r"P(n \mid \lambda) = \frac{\lambda^n e^{-\lambda}}{n!}")
        st.markdown('A Gaussian random variable has the probability distribution function')
        st.latex(r"""
        P(x \mid \mu, \sigma) =
        \frac{1}{\sqrt{2\pi\sigma^2}}
        \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
        """)
        st.markdown("Given multiple independent data $\\{x_i\\}$ and parameters $\\theta$ the likelihood of observing some data given the parameters $\theta$ is:")

        st.latex(r"\mathcal{L}(\theta) = \prod_{i} P(x_i \mid \theta)")

        st.markdown("We deal with the log likelihood:")

        st.latex(r"\log \mathcal{L}(\theta) = \sum_i \log P(x_i \mid \theta)")

        st.markdown("We estimate parameters by maximizing likelihood:")

        st.latex(r"\hat{\theta} = \arg\max_{\theta} \log \mathcal{L}(\theta)")

        st.markdown("For Gaussian noise with variance $\\sigma^2$:")
        st.latex(r"""
        \log \mathcal{L}
        = -\frac{1}{2}
        \sum_i
        \left[
        \frac{(x_i - f_i(\theta))^2}{\sigma^2}
        + \log(2\pi\sigma^2)
        \right]
        """)

        st.info("This is equivalent to **least-squares minimization**.")
        st.markdown("#### Parameter Uncertainty")

        st.markdown(r'One usually sees parameter uncertainty as $\mu \pm \sigma$. This comes from a **Gaussian** approximation of the likelihood.')
        st.markdown(r'This is calculated by first calculating the Fisher Information Matrix $\mathcal{I}$:')
        st.latex(r"""
        \mathcal{I}_{ij}
        = -\mathbb{E}
        \left[
        \frac{\partial^2 \log \mathcal{L}}
        {\partial \theta_i \partial \theta_j}
        \right]
        """)
        st.markdown('The Fisher information matrix inverse gives the covariance')
        st.latex(r"\Sigma(\theta) = \mathcal{I}^{-1}")

        st.markdown("Near the maximum:")

        st.latex(r"""
        \log \mathcal{L}(\theta)
        \approx
        \log \mathcal{L}(\hat{\theta})
        -
        \frac{1}{2}
        (\theta - \hat{\theta})^T
        \mathcal{I}
        (\theta - \hat{\theta})
        """)
        st.markdown("Contours of constant likelihood form **ellipses**.")

# ============================================================
# Run the app
# ============================================================
if __name__ == "__main__":
    main()
