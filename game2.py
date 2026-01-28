import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.stats import poisson
import pandas as pd
import matplotlib as mpl
import matplotlib.gridspec 

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
# TRUE Parameter Values (Known a-priori for simulation)
# ============================================================
TRUE_AMP0 = 2.0
TRUE_E00 = 5.0
TRUE_BG0 = 1.5
TRUE_HWHM0 = 0.15

TRUE_T1_NON_PRESSURIZED = 50.0
TRUE_T2_NON_PRESSURIZED = 150.0
TRUE_SLOPE1_NON_PRESSURIZED = 0.1 / 48.0
TRUE_SLOPE2_NON_PRESSURIZED = 0.8 / 100.0
TRUE_SLOPE3_NON_PRESSURIZED = 0.3 / 120.0

TRUE_T1_PRESSURIZED = 50.0
TRUE_T2_PRESSURIZED = 150.0
TRUE_SLOPE1_PRESSURIZED = 0.2 / 48.0
TRUE_SLOPE2_PRESSURIZED = 1.2 / 100.0
TRUE_SLOPE3_PRESSURIZED = 0.5 / 120.0

# ============================================================
# Game Settings
# ============================================================
TIME_BUDGET = 12 * 3600  # 12 hours in seconds

def get_empty_scans_df():
    """Return an empty DataFrame with the correct schema"""
    return pd.DataFrame({
        'T': pd.Series(dtype='float64'),
        'Energy': pd.Series(dtype='float64'),
        'counts_per_sec': pd.Series(dtype='float64'),
        'error_l': pd.Series(dtype='float64'),
        'error_h': pd.Series(dtype='float64'),
        'lam': pd.Series(dtype='float64'),
        'amp': pd.Series(dtype='float64'),
        'E0': pd.Series(dtype='float64'),
        'hwhm': pd.Series(dtype='float64'),
        'bg': pd.Series(dtype='float64'),
        'counts': pd.Series(dtype='int64'),
        'count_time': pd.Series(dtype='float64'),
        'pressurized': pd.Series(dtype='bool'),
    })

# ============================================================
# JAX Warmup Function
# ============================================================
def warmup_jax():
    """Warm up JAX compilations to avoid first-run slowdowns"""
    if st.session_state.jax_warmed_up:
        return
    
    # Dummy scan size (representative, not huge)
    N = 30

    energy = jnp.linspace(2.0, 8.0, N)
    temp = jnp.ones(N) * 50.0
    pressurized = jnp.zeros(N, dtype=bool)
    count_time = jnp.ones(N) * 60.0

    theta0 = jnp.array([
        TRUE_SLOPE1_PRESSURIZED,
        TRUE_SLOPE2_PRESSURIZED,
        TRUE_SLOPE3_PRESSURIZED,
        TRUE_SLOPE1_NON_PRESSURIZED,
        TRUE_SLOPE2_NON_PRESSURIZED,
        TRUE_SLOPE3_NON_PRESSURIZED,
    ])

    # Compile model
    _ = meta_model(energy[0], temp[0], pressurized[0], *theta0)

    # Compile lambda_model
    _ = lambda_model(theta0, energy, temp, pressurized)

    # Compile Fisher Jacobian
    _ = lambda_model_jacobian(theta0, energy, temp, pressurized)

    # Compile HWHM Jacobian
    args = (
        energy,
        temp,
        jnp.ones(N) * TRUE_AMP0,
        jnp.ones(N) * TRUE_E00,
        jnp.ones(N) * TRUE_HWHM0,
        jnp.ones(N) * TRUE_BG0,
    )
    _ = model(*args)

    _ = temperature_dependent_hwhm(temp[0], args[4], pressurized=False, 
                               T1=50, T2=150, 
                               slope1p=TRUE_SLOPE1_PRESSURIZED, slope2p=TRUE_SLOPE2_PRESSURIZED, slope3p=TRUE_SLOPE3_PRESSURIZED,
                               slope1np=TRUE_SLOPE1_NON_PRESSURIZED, slope2np=TRUE_SLOPE2_NON_PRESSURIZED, slope3np=TRUE_SLOPE3_NON_PRESSURIZED)
    
    st.session_state.jax_warmed_up = True

# ============================================================
# Damped Harmonic Oscillator Model
# ============================================================
def bose(E, T):
    """Vectorized Bose factor calculation"""
    abs_E = jnp.abs(E)
    bose_factor = 1.0 / (jnp.exp(abs_E / (kB * T)) - 1.0)
    return jnp.where(E >= 0, bose_factor + 1.0, bose_factor)

def DampedHarmonicOscillator(E, T, E0, hwhm, amp):
    """Damped harmonic oscillator with proper Bose factor"""
    symmetric_part = jnp.abs(amp / (E0 * jnp.pi) *
        (hwhm / ((E - E0)**2 + hwhm**2) - hwhm / ((E + E0)**2 + hwhm**2)))
    return symmetric_part * bose(E, T)

@jax.jit
def model(x, T, amp, E0, hwhm, bg):
    """Full model: Damped Harmonic Oscillator with Bose factor + background"""
    return DampedHarmonicOscillator(x, T, E0, hwhm, amp) + bg

@jax.jit
def smooth_step(x, x0, width):
    """Smooth transition from 0 to 1"""
    return 0.5 * (1.0 + jnp.tanh((x - x0) / width))

@jax.jit
def select_slopes(pressurized,
                  slope1p, slope2p, slope3p,
                  slope1np, slope2np, slope3np):
    pressurized = jnp.asarray(pressurized)
    slopes_p  = jnp.array([slope1p,  slope2p,  slope3p])
    slopes_np = jnp.array([slope1np, slope2np, slope3np])
    return jnp.where(pressurized, slopes_p, slopes_np)

@jax.jit
def temperature_dependent_hwhm(
    T,
    hwhm0,
    pressurized,
    T1=50.0,
    T2=150.0,
    Tmin=2.0,
    transition_width=8.0,
    slope1p=TRUE_SLOPE1_PRESSURIZED,
    slope2p=TRUE_SLOPE2_PRESSURIZED,
    slope3p=TRUE_SLOPE3_PRESSURIZED,
    slope1np=TRUE_SLOPE1_NON_PRESSURIZED,
    slope2np=TRUE_SLOPE2_NON_PRESSURIZED,
    slope3np=TRUE_SLOPE3_NON_PRESSURIZED,
):
    """
    Smooth, differentiable, piecewise-linear HWHM(T)
    """
    T = jnp.asarray(T)
    pressurized = jnp.asarray(pressurized)

    m1, m2, m3 = select_slopes(
        pressurized,
        slope1p, slope2p, slope3p,
        slope1np, slope2np, slope3np,
    )

    # Linear segments (anchored correctly)
    h1 = hwhm0 * (1.0 + m1 * (T - Tmin))
    h1_T1 = hwhm0 * (1.0 + m1 * (T1 - Tmin))

    h2 = h1_T1 + hwhm0 * m2 * (T - T1)
    h2_T2 = h1_T1 + hwhm0 * m2 * (T2 - T1)

    h3 = h2_T2 + hwhm0 * m3 * (T - T2)

    # Smooth weights
    s1 = smooth_step(T, T1, transition_width)
    s2 = smooth_step(T, T2, transition_width)

    w1 = 1.0 - s1
    w2 = s1 * (1.0 - s2)
    w3 = s2

    return w1 * h1 + w2 * h2 + w3 * h3

def temperature_dependent_params(T, amp0, hwhm0, bg0, E00, pressurized=False):
    """Temperature dependence for all parameters"""
    amp = amp0 * (1.0 - 0.3 * ((T - 2.0) / 268.0))
    bg = bg0 * (1.0 + 2.5 * ((T - 2.0) / 268.0))
    E0 = E00 * (1.0 - 0.1 * ((T - 2.0) / 268.0))
    
    hwhm = temperature_dependent_hwhm(T, hwhm0, pressurized)
    
    return amp, E0, hwhm, bg

def sample_poisson_scan(T, amp0, hwhm0, bg0, E00, npts, E_range, count_time, scan_center, pressurized=False):
    """Sample with temperature-dependent parameters and return normalized counts"""
    amp, E0, hwhm, bg = temperature_dependent_params(T, amp0, hwhm0, bg0, E00, pressurized)
    
    Es = np.linspace(scan_center - E_range/2, scan_center + E_range/2, npts)
    Es_jax = jnp.array(Es)
    lam = np.array(model(Es_jax, T, amp, E0, hwhm, bg))
    
    counts = np.random.poisson(lam * count_time)
    counts_per_sec = counts / count_time
    
    alpha = 0.32
    lower = poisson.ppf(alpha/2, counts) / count_time
    upper = poisson.ppf(1 - alpha/2, counts) / count_time
    errors = np.vstack([counts_per_sec - lower, upper - counts_per_sec])
    
    return Es, counts_per_sec, errors, lam, amp, E0, hwhm, bg, counts

@jax.jit
def meta_model(E, T, pressurized,
               slope1p, slope2p, slope3p,
               slope1np, slope2np, slope3np):
    amp = TRUE_AMP0 * (1.0 - 0.3 * ((T - 2.0) / 268.0))
    bg  = TRUE_BG0  * (1.0 + 2.5 * ((T - 2.0) / 268.0))
    E0  = TRUE_E00  * (1.0 - 0.1 * ((T - 2.0) / 268.0))

    hwhm = temperature_dependent_hwhm(
        T, TRUE_HWHM0,
        pressurized=jnp.asarray(pressurized),
        slope1p=slope1p, slope2p=slope2p, slope3p=slope3p,
        slope1np=slope1np, slope2np=slope2np, slope3np=slope3np,
    )

    return model(E, T, amp, E0, hwhm, bg)

@jax.jit
def lambda_model(theta, E, T, pressurized):
    s1p, s2p, s3p, s1np, s2np, s3np = theta
    rate = jax.vmap(
        meta_model,
        in_axes=(0, 0, 0, None, None, None, None, None, None)
    )(E, T, pressurized,
      s1p, s2p, s3p, s1np, s2np, s3np)
    return rate

@jax.jit
def lambda_model_jacobian(theta0, energy, temp, pressurized):
    return jax.jacobian(lambda_model, argnums=0)(theta0, energy, temp, pressurized)

def compute_fisher_scores():
    if len(st.session_state.all_scans) == 0:
        return -np.inf, -np.inf, -np.inf

    # Extract arrays
    E = jnp.array(st.session_state.all_scans['Energy'].values)
    T = jnp.array(st.session_state.all_scans['T'].values)
    pressurized = jnp.array(st.session_state.all_scans['pressurized'].values)
    count_time = jnp.array(st.session_state.all_scans['count_time'].values)

    theta0 = jnp.array([
        TRUE_SLOPE1_PRESSURIZED,
        TRUE_SLOPE2_PRESSURIZED,
        TRUE_SLOPE3_PRESSURIZED,
        TRUE_SLOPE1_NON_PRESSURIZED,
        TRUE_SLOPE2_NON_PRESSURIZED,
        TRUE_SLOPE3_NON_PRESSURIZED,
    ])

    # λ_i
    lam = lambda_model(theta0, E, T, pressurized)
    lam = jnp.clip(lam, 1e-12, jnp.inf)

    # Jacobian
    J = lambda_model_jacobian(theta0, E, T, pressurized)

    J_pressurized = J[:,:3]
    J_non_pressurized = J[:,3:]

    lp = fisher_subfunc(J_pressurized, lam, count_time)
    lnp = fisher_subfunc(J_non_pressurized, lam, count_time)
    
    return float(lnp), float(lp), float(lnp + lp)

@jax.jit
def fisher_subfunc(J, lam, count_time):
    def logdet(mat):
        sign, ld = jnp.linalg.slogdet(mat)
        return jnp.where(sign > 0, ld, -jnp.inf)

    return logdet(J.T @ (J * (count_time / lam)[:, None]))

# @jax.jit
def compute_hwhm_fisher(df_subset):
    """Compute Fisher information for HWHM at specific temperature"""
    if len(df_subset) == 0:
        return np.inf  # Infinite error if no data
    
    # Extract data for this temperature group
    E = jnp.array(df_subset['Energy'].values)
    T_val = df_subset['T'].iloc[0]
    pressurized = df_subset['pressurized'].iloc[0]
    count_time = jnp.array(df_subset['count_time'].values)
    
    # Get true parameters at this temperature
    amp_true = df_subset['amp'].iloc[0]
    E0_true = df_subset['E0'].iloc[0]
    hwhm_true = df_subset['hwhm'].iloc[0]
    bg_true = df_subset['bg'].iloc[0]
    
    # Compute derivative of model with respect to HWHM
    @jax.jit
    def model_hwhm_derivative(E, T, amp, E0, hwhm, bg):
        """Compute d(model)/d(hwhm)"""
        # Helper function to compute the derivative
        def dho_dhwhm(E, T, E0, hwhm, amp):
            # Derivative of the Damped Harmonic Oscillator with respect to hwhm
            term1 = hwhm / ((E - E0)**2 + hwhm**2)
            term2 = hwhm / ((E + E0)**2 + hwhm**2)
            
            # Derivative of term1 with respect to hwhm
            dterm1 = ((E - E0)**2 - hwhm**2) / ((E - E0)**2 + hwhm**2)**2
            dterm2 = ((E + E0)**2 - hwhm**2) / ((E + E0)**2 + hwhm**2)**2
            
            return (amp / (E0 * jnp.pi)) * (dterm1 - dterm2) * bose(E, T)
        
        return dho_dhwhm(E, T, E0, hwhm, amp)
    
    # Vectorize the derivative computation
    compute_derivatives = jax.vmap(model_hwhm_derivative, in_axes=(0, None, None, None, None, None))
    
    # Compute derivatives for all energy points
    derivatives = compute_derivatives(E, T_val, amp_true, E0_true, hwhm_true, bg_true)
    
    # Compute predicted counts
    compute_counts = jax.vmap(model, in_axes=(0, None, None, None, None, None))
    lam = compute_counts(E, T_val, amp_true, E0_true, hwhm_true, bg_true)
    lam = jnp.clip(lam, 1e-12, jnp.inf)
    
    # Compute Fisher information for HWHM
    # For Poisson statistics: Fisher information = Σ_i (dλ_i/dθ)² * (t_i/λ_i)
    fisher_info = jnp.sum((derivatives**2) * (count_time / lam))
    
    # Standard error = 1/sqrt(Fisher information)
    # Add small regularization to avoid division by zero
    fisher_info = jnp.maximum(fisher_info, 1e-12)
    std_error = 1.0 / jnp.sqrt(fisher_info)
    
    return float(std_error)

# Replace the data summary section (around line 800-850) with:
def compute_total_fisher_matrix_all_params():
    """Compute and return the full Fisher information matrix for ALL parameters"""
    if len(st.session_state.all_scans) == 0:
        return None, None
    
    # Extract arrays
    E = jnp.array(st.session_state.all_scans['Energy'].values)
    T = jnp.array(st.session_state.all_scans['T'].values)
    pressurized = jnp.array(st.session_state.all_scans['pressurized'].values)
    count_time = jnp.array(st.session_state.all_scans['count_time'].values)
    
    # Define full parameter set (10 parameters)
    theta_full = jnp.array([
        TRUE_AMP0,               # 0: Amplitude at 2K
        TRUE_E00,                # 1: E0 at 2K  
        TRUE_BG0,                # 2: Background at 2K
        TRUE_HWHM0,              # 3: HWHM intercept at 2K
        TRUE_SLOPE1_PRESSURIZED,     # 4: Pressurized slope 1 (T < 50K)
        TRUE_SLOPE2_PRESSURIZED,     # 5: Pressurized slope 2 (50K < T < 150K)
        TRUE_SLOPE3_PRESSURIZED,     # 6: Pressurized slope 3 (T > 150K)
        TRUE_SLOPE1_NON_PRESSURIZED, # 7: Non-pressurized slope 1
        TRUE_SLOPE2_NON_PRESSURIZED, # 8: Non-pressurized slope 2  
        TRUE_SLOPE3_NON_PRESSURIZED, # 9: Non-pressurized slope 3
    ])
    
    # Define the full model function with all parameters
    @jax.jit
    def full_model_single(E, T, pressurized, *params):
        amp0, E00, bg0, hwhm0, s1p, s2p, s3p, s1np, s2np, s3np = params
        
        # Temperature-dependent parameters
        amp = amp0 * (1.0 - 0.3 * ((T - 2.0) / 268.0))
        bg = bg0 * (1.0 + 2.5 * ((T - 2.0) / 268.0))
        E0 = E00 * (1.0 - 0.1 * ((T - 2.0) / 268.0))
        
        hwhm = temperature_dependent_hwhm(
            T, hwhm0,
            pressurized=jnp.asarray(pressurized),
            slope1p=s1p, slope2p=s2p, slope3p=s3p,
            slope1np=s1np, slope2np=s2np, slope3np=s3np,
        )
        
        return model(E, T, amp, E0, hwhm, bg)
    
    # Vectorized model
    @jax.jit
    def lambda_model_full(theta, E, T, pressurized):
        return jax.vmap(full_model_single, in_axes=(0, 0, 0, *(None for _ in range(10))))(
            E, T, pressurized, *theta
        )
    
    # Compute Jacobian
    @jax.jit
    def compute_jacobian_full(theta, E, T, pressurized):
        return jax.jacobian(lambda_model_full, argnums=0)(theta, E, T, pressurized)
    
    # Compute λ_i
    lam = lambda_model_full(theta_full, E, T, pressurized)
    lam = jnp.clip(lam, 1e-12, jnp.inf)
    
    # Jacobian
    J_full = compute_jacobian_full(theta_full, E, T, pressurized)
    
    # Fisher information matrix
    W = count_time / lam
    FI_full = J_full.T @ (J_full * W[:, None])
    
    # Parameter names in English
    param_names_full = [
        "Amplitude (2K)",
        "Peak Center E0 (2K)", 
        "Background (2K)",
        "HWHM Intercept (2K)",
        "Pressurized Slope 1 (T < 50K)",
        "Pressurized Slope 2 (50-150K)",
        "Pressurized Slope 3 (T > 150K)",
        "Non-Pressurized Slope 1 (T < 50K)",
        "Non-Pressurized Slope 2 (50-150K)",
        "Non-Pressurized Slope 3 (T > 150K)"
    ]
    
    return np.array(FI_full), param_names_full


def compute_schur_complement():
    """Compute Schur complement for slope parameters after conditioning out nuisance parameters"""
    FI_full, param_names = compute_total_fisher_matrix_all_params()
    if FI_full is None:
        return None, None, None
    
    # Partition indices
    # Nuisance parameters: amplitude, E0, background, HWHM intercept (indices 0-3)
    # Slope parameters: all slopes (indices 4-9)
    n_nuisance = 4
    n_slopes = 6
    
    FI_AA = FI_full[:n_nuisance, :n_nuisance]
    FI_AB = FI_full[:n_nuisance, n_nuisance:]
    FI_BA = FI_full[n_nuisance:, :n_nuisance]
    FI_BB = FI_full[n_nuisance:, n_nuisance:]
    
    # Compute Schur complement
    try:
        FI_AA_inv = np.linalg.inv(FI_AA + np.eye(n_nuisance) * 1e-10)  # Regularize
        schur_complement = FI_BB - FI_BA @ FI_AA_inv @ FI_AB
        
        # Slope parameter names
        slope_names = param_names[n_nuisance:]
        
        # Partition into pressurized and non-pressurized
        schur_press = schur_complement[:3, :3]
        schur_non_press = schur_complement[3:, 3:]
        
        # Compute log determinants
        def logdet_safe(mat):
            sign, ld = np.linalg.slogdet(mat)
            return ld if sign > 0 else -np.inf
        
        score_press = logdet_safe(schur_press)
        score_non_press = logdet_safe(schur_non_press)
        score_total = logdet_safe(schur_complement)
        
        return score_non_press, score_press, score_total, schur_complement, slope_names
        
    except np.linalg.LinAlgError:
        return -np.inf, -np.inf, -np.inf, None, None



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
    Es, counts_per_sec, errors, lam, amp, E0, hwhm, bg, counts = sample_poisson_scan(
        T, TRUE_AMP0, TRUE_HWHM0, TRUE_BG0, TRUE_E00, npts, E_range, count_time, scan_center, pressurized
    )
    
    
    new_df = pd.DataFrame({
        'T': T,
        'Energy': Es,
        'counts_per_sec': counts_per_sec,
        'error_l': errors.T[:, 0],
        'error_h': errors.T[:, 1],
        'lam': lam,
        'amp': amp,
        'E0': E0,
        'hwhm': float(hwhm),
        'bg': bg,
        'counts': counts,
        'count_time': count_time,
        'pressurized': pressurized,
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
                hwhm_err = compute_hwhm_fisher(df)
                ax3.errorbar(T, df['hwhm'].mean(), hwhm_err,
                            linestyle='none', color='gold', linewidth=2,
                            marker='o', markersize=6, markeredgecolor='gold',
                            markeredgewidth=0.5, capsize=4, capthick=2,
                            label='Non-Pressurized' if T == non_press_scans['T'].min() else None)
        
        # Pressurized measurements
        press_scans = st.session_state.all_scans[st.session_state.all_scans['pressurized']]
        for T, df in press_scans.groupby('T'):
            if len(df) > 0:
                hwhm_err = compute_hwhm_fisher(df)
                ax3.errorbar(T, df['hwhm'].mean(), hwhm_err,
                            linestyle='none', color='red', linewidth=2,
                            marker='o', markersize=6, markeredgecolor='red',
                            markeredgewidth=0.5, capsize=4, capthick=2,
                            label='Pressurized' if T == press_scans['T'].min() else None)
        
    # Theoretical guides
    if st.session_state.show_theory:
        T_guide = np.linspace(2, 270, 100)
        hwhm_guide_non = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, False) for T_val in T_guide]
        hwhm_guide_press = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, True) for T_val in T_guide]
        
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
        warmup_jax()
    
    st.markdown('<h1 class="main-header">Neutron Scattering Experiment Simulator</h1>', unsafe_allow_html=True)
    
    # Header info
    st.markdown("""
**Problem:** You are an experimentalist with 2 samples of the same crystal. In one sample, uniaxial pressure is applied. In the other, there is no pressure applied.
                
**Goal:** Your goal is to understand how the width of the lineshape changes with increasing temperature.
                
**Resources:** You have 12 hours and can control the experiment using the widgets on the left.
                
**Scoring:** You are scored based on the precision of the parameters fit.

""")
    st.info('**Note:** The score is based on the precision of the **slope** of the half-width at half max as a function of temperature. '
    ' This is a piecewise linear model, so energy scans at different temperatures are useful. Also, all parameters are treated' \
    'as already fit, which is not realistic (but keeps the game able to be computed quickly)')    
    # Sidebar for controls
    st.markdown("---")
    st.markdown("### How to Play")
    st.info("""
    1. Adjust scan parameters in the sidebar
    2. Run scans at different temperatures
    3. Balance between pressurized and non-pressurized conditions
    4. Maximize your Fisher score within 12 hours
    5. Try to cover a wide temperature range!
    """)
    with st.sidebar:
        st.markdown("## Experiment Controls")
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
    st.markdown("### Data Summary")
    if len(st.session_state.all_scans) > 0:
        # Compute Fisher information for all parameters
        FI_full, param_names_full = compute_total_fisher_matrix_all_params()
        
        # Compute scores using Schur complement
        schur_non_score, schur_press_score, schur_total_score, schur_matrix, slope_names = compute_schur_complement()
        
        def format_score(score):
            if score == -np.inf or np.isnan(score):
                return "-∞"
            else:
                return f"{score:.2f}"
        
        st.markdown("### Fisher Information Scores (Schur Complement)")
        st.markdown(f'##### Total Schur Score: {format_score(schur_total_score)}')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'Non-Pressurized Schur Score: {format_score(schur_non_score)}')
        with col2:
            st.markdown(f'Pressurized Schur Score: {format_score(schur_press_score)}')
        
        st.info("""
        **Schur Complement Score**: Measures information about slope parameters 
        AFTER accounting for uncertainty in amplitude, peak position, background, and HWHM intercept.
        Higher score = better ability to determine temperature dependence.
        """)

        # Display Full Fisher Information Matrix
        with st.expander('Matrix Visualization'):
            if FI_full is not None:
                tab1, tab2, tab3 = st.tabs(["Fisher Matrix", "Covariance Matrix", "Schur Complement"])
                
                with tab1:
                    # Create a more readable display
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Use shorter names for display
                    short_names = [
                        "Amp(2K)", "E0(2K)", "BG(2K)", "HWHM(2K)",
                        "P Slope1", "P Slope2", "P Slope3",
                        "NP Slope1", "NP Slope2", "NP Slope3"
                    ]
                    
                    # Normalize for better visualization
                    FI_norm = np.abs(FI_full) / np.max(np.abs(FI_full))
                    
                    im = ax.imshow(FI_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
                    
                    # Set labels
                    ax.set_xticks(range(10))
                    ax.set_yticks(range(10))
                    ax.set_xticklabels(short_names, rotation=45, ha='right')
                    ax.set_yticklabels(short_names)
                    ax.set_title("Normalized Fisher Information Matrix (|values|/max)")
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Normalized Information')
                    
                    # Add text annotations for matrix values (only significant ones)
                    threshold = 0.1  # Only show values above 10% of max
                    for i in range(10):
                        for j in range(10):
                            if np.abs(FI_full[i, j]) > np.max(np.abs(FI_full)) * threshold:
                                text = ax.text(j, i, f'{FI_full[i, j]:.1e}',
                                            ha="center", va="center", 
                                            color="white" if FI_norm[i, j] > 0.5 else "black",
                                            fontsize=7)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show interpretation
                    with st.expander("How to interpret this matrix"):
                        st.markdown("""
                        **Color Coding:**
                        - 🔴 **Red**: Strong information (parameters well-constrained)
                        - ⚪ **White**: Moderate information  
                        - 🔵 **Blue**: Weak information (parameters poorly constrained)
                        
                        **Matrix Structure:**
                        - **Diagonal elements**: How well each parameter is determined individually
                        - **Off-diagonal elements**: How parameters are correlated
                        - **Blocks**: Related parameters (e.g., slopes) often form blocks
                        
                        **What makes a good matrix?**
                        1. **Strong diagonal** = Each parameter is individually measurable
                        2. **Weak off-diagonals** = Parameters are independent
                        3. **Well-conditioned** = All eigenvalues are positive and not too different
                        """)
                
                with tab2:
                    # with col2:
                    st.markdown("#### Covariance Matrix")
                    # Rank parameters by their diagonal values (individual information)
                    
                    # Compute correlations from covariance matrix
                    try:
                        covariance = np.linalg.pinv(FI_full + np.eye(10) * 1e-10)
                             # Create a more readable display
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Use shorter names for display
                        short_names = [
                            "Amp(2K)", "E0(2K)", "BG(2K)", "HWHM(2K)",
                            "P Slope1", "P Slope2", "P Slope3",
                            "NP Slope1", "NP Slope2", "NP Slope3"
                        ]
                        
                        # Normalize for better visualization
                        
                        im = ax.imshow(covariance, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
                        
                        # Set labels
                        ax.set_xticks(range(10))
                        ax.set_yticks(range(10))
                        ax.set_xticklabels(short_names, rotation=45, ha='right')
                        ax.set_yticklabels(short_names)
                        ax.set_title("Covariance Matrix")
                        
                        # Add colorbar
                        plt.colorbar(im, ax=ax, label='Covariane')
                        
                        # Add text annotations for matrix values (only significant ones)
                        threshold = 0.1  # Only show values above 10% of max
                        for i in range(10):
                            for j in range(i+1):
                                text = ax.text(j, i, f'{covariance[i, j]:.1e}',
                                            ha="center", va="center", 
                                            color="black",
                                            fontsize=7)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except np.linalg.LinAlgError:
                        st.warning("Cannot compute correlations - matrix is singular")
                
                with tab3:
                    # Display Schur Complement Matrix
                    st.markdown("#### Schur Complement Matrix (Slopes Only)")
                    st.markdown("Information about slope parameters AFTER accounting for uncertainty in other parameters")
                    
                    if schur_matrix is not None:
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        
                        # Normalize for visualization
                        schur_norm = np.abs(schur_matrix) / np.max(np.abs(schur_matrix))
                        
                        im2 = ax2.imshow(schur_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
                        
                        # Short names for slopes
                        slope_short_names = ["P S1", "P S2", "P S3", "NP S1", "NP S2", "NP S3"]
                        
                        ax2.set_xticks(range(6))
                        ax2.set_yticks(range(6))
                        ax2.set_xticklabels(slope_short_names, rotation=45, ha='right')
                        ax2.set_yticklabels(slope_short_names)
                        ax2.set_title("Schur Complement Matrix")
                        
                        plt.colorbar(im2, ax=ax2, label='Normalized Information')
                        
                        # Add text annotations
                        for i in range(6):
                            for j in range(6):
                                if np.abs(schur_matrix[i, j]) > np.max(np.abs(schur_matrix)) * 0.1:
                                    text = ax2.text(j, i, f'{schur_matrix[i, j]:.1e}',
                                                ha="center", va="center", 
                                                color="white" if schur_norm[i, j] > 0.5 else "black",
                                                fontsize=8)
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
                        # Schur complement statistics
                        st.markdown("#### Schur Complement Statistics")
                        
                        schur_eigenvalues = np.linalg.eigvals(schur_matrix)
                        pos_schur_eigenvalues = schur_eigenvalues[schur_eigenvalues.real > 0]
                        
                        if len(pos_schur_eigenvalues) > 0:
                            schur_cond = np.max(np.abs(schur_eigenvalues)) / np.min(np.abs(pos_schur_eigenvalues))
                        else:
                            schur_cond = np.inf
                        
                        schur_stats = {
                            "Total Information (Trace)": f"{np.trace(schur_matrix):.2e}",
                            "Determinant": f"{np.linalg.det(schur_matrix):.2e}",
                            "Condition Number": f"{schur_cond:.2e}",
                            "Information Loss to Nuisance Params": f"{100*(1 - np.trace(schur_matrix)/np.trace(FI_full[4:, 4:])):.1f}%",
                        }
                        
                        for key, value in schur_stats.items():
                            st.metric(key, value)
                        
                        st.info("""
                        **Information Loss**: Shows how much information about slopes is "lost" because 
                        we also have to determine amplitude, peak position, background, and HWHM intercept.
                        """)
            
        # Traditional summary data
        st.markdown("### Experimental Summary")
        
        summary_data = {
            "Total # of Measurements": len(st.session_state.all_scans),
            "Non-Pressurized Measurements": len(st.session_state.all_scans[~st.session_state.all_scans['pressurized']]),
            "Pressurized Measurements": len(st.session_state.all_scans[st.session_state.all_scans['pressurized']]),
            "Unique Temperatures": st.session_state.all_scans['T'].nunique(),
            "Temperature Range Covered": f"{st.session_state.all_scans['T'].min():.1f}K to {st.session_state.all_scans['T'].max():.1f}K",
            "Total Measurement Time": f"{st.session_state.used_time/3600:.2f} hours",
            "Remaining Time": f"{max(0, (TIME_BUDGET - st.session_state.used_time)/3600):.2f} hours",
            "Time Used": f"{min(100, (st.session_state.used_time/TIME_BUDGET)*100):.1f}%"
        }
        
        cols = st.columns(3)
        for idx, (key, value) in enumerate(summary_data.items()):
            cols[idx % 3].metric(key, value)
        

    st.markdown('---')
    st.markdown('## Understanding the Game')

    with st.expander("### The Physics Experiment"):
        st.markdown("#### Neutron Scattering Model")

        st.latex(r"""
        I(E, T) = S(E, T) + B(T)
        """)

        st.markdown("The scattering function is modeled as a damped harmonic oscillator:")
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

        st.markdown("Piecewise-linear segments:")
        st.latex(r"""
        \Gamma_1(T) = \Gamma_0 \left[ 1 + m_1 (T - T_{\min}) \right]
        """)
        st.latex(r"""
        \Gamma_2(T) = \Gamma_1(T_1) + \Gamma_0 m_2 (T - T_1)
        """)
        st.latex(r"""
        \Gamma_3(T) = \Gamma_2(T_2) + \Gamma_0 m_3 (T - T_2)
        """)

        st.markdown("Smooth transition weights:")
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

        st.markdown("#### Measurement Statistics")

        st.latex(r"""
        N_i \sim \mathrm{Poisson}\!\left( I(E_i, T) \cdot t_i \right)
        """)

        st.markdown(
            "Each energy point is measured with Poisson counting statistics, "
            "which determine the uncertainties and the Fisher information used for scoring."
        )

    with st.expander('### Experimental Trade-offs'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
    **Time Management:**
    - Each measurement costs time
    - 12 hours total = limited resource
    - Need to choose: few precise measurements OR many quick ones

    **Counting Statistics:**
    - Longer count time → smaller error bars
    - But fewer total measurements
    - Poisson statistics: Error ≈ √(counts)

    **Temperature Coverage:**
    - Need measurements at different temperatures to see trends
    - But each temperature scan takes time
    - Critical temperatures: 50K, 150K (transition points)
    """)
        
        with col2:
            st.markdown("""
    **Energy Range Choices:**
    - Wide range: See full peak shape + background
    - Narrow range: Focus on peak region, less background
    - Center position: Must include the actual peak!

    **Points vs. Time:**
    - More points: Better energy resolution
    - But each point takes counting time
    - Fewer points: Quicker scans, but might miss details

    **Balance Strategy:**
    1. Quick scans to find peaks
    2. Medium scans to map temperature dependence
    3. Long scans at key temperatures for precision
    """)

    with st.expander('### Measurement Statistics 101'):
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

    with st.expander('### Fisher Information Score Explained'):
        st.markdown("""
    **What is Fisher Information?**
    A mathematical measure of how much your measurements tell you about the parameters you care about (the slopes of HWHM vs temperature).

    **How is it calculated?**
    For each measurement point, we compute:

    $$\\text{Fisher contribution} = \\frac{\\text{count time}}{\\text{count rate}} \\times \\left(\\frac{\\partial\\text{count rate}}{\\partial\\text{parameter}}\\right)^2$$

    Then we sum over all measurements.

    **What does your score mean?**
    - **Higher score** = Better precision in your parameter estimates
    - **Negative infinity (-∞)** = Not enough data to determine all parameters
    - **Separate scores** for pressurized vs. non-pressurized samples

    **How to improve your score:**
    1. Measure where the signal changes most with temperature (near transition points)
    2. Get good statistics at key temperatures
    3. Measure both pressurized and non-pressurized
    4. Cover the full temperature range (2K to 270K)
    """)
        

    with st.expander('### Bayesian vs Frequentist Thinking'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
    **Frequentist Approach (What this game uses):**
    - Parameters have fixed true values
    - Uncertainty comes from measurement randomness
    - "95% confidence interval": If we repeated experiment 100 times, 95 intervals would contain true value
    - Uses Fisher Information to quantify precision
    """)
        
        with col2:
            st.markdown("""
    **Bayesian Approach (Alternative view):**
    - Parameters have probability distributions
    - Start with prior belief, update with data
    - "95% credible interval": 95% probability true value is in interval
    - Incorporates prior knowledge explicitly

    **Connection:**
    With little data: Frequentist and Bayesian can differ
    With lots of data: They must agree
    """)

    with st.expander('### Optimal Experimental Design'):
        st.markdown("""
    **The Big Idea:**
    Instead of measuring randomly, choose measurements that give you the most information!

    **Information-Rich Measurements:**
    1. **Where model is sensitive**: Measure where count rate changes a lot when parameters change
    2. **Where uncertainty is high**: Focus on regions with currently poor constraints
    3. **Where it matters most**: For HWHM, measure near the peak edges

    **Real-World Analogy:**
    Imagine trying to map a mountain:
    - **Bad strategy**: Measure everywhere equally
    - **Good strategy**: Focus on the slopes (where elevation changes)
    - **Best strategy**: Measure slopes AND connect different sides

    **In This Game:**
    The Fisher score tells you how well you're doing. Try different strategies:
    - Many temperatures with few points?
    - Few temperatures with many points?
    - Mix of both?
    """)

    with st.expander('### Practical Tips for Success'):
        st.markdown("""
    **Getting Started:**
    1. **Quick reconnaissance**: Do fast scans at 50K, 150K, 250K for both samples
    2. **Find the peaks**: Adjust energy center to capture the peak
    3. **Check theory lines**: Use them as guides (toggle on/off with button)

    **Intermediate Strategy:**
    4. **Identify key regions**: Where does HWHM change most? Focus there!
    5. **Balance samples**: Don't neglect one sample type
    6. **Watch transition points**: 50K and 150K are important

    **Advanced Optimization:**
    7. **Check Fisher score often**: Are you making progress?
    8. **Adjust based on results**: If score is -∞, need more data
    9. **Time management**: Leave buffer at the end for final precision scans

    **Common Pitfalls:**
    - ⚠️ **Too few temperatures**: Can't see temperature dependence
    - ⚠️ **Too many points per scan**: Wastes time, fewer temperatures
    - ⚠️ **Wrong energy range**: Missing the peak entirely
    - ⚠️ **Ignoring one sample**: Incomplete comparison
    """)

    with st.expander('### Real-World Applications'):
        st.markdown("""
    **Why This Matters in Real Science:**

    **1. Neutron Scattering Facilities:**
    - Cost millions to build and operate
    - Beam time is precious (hours cost thousands of dollars)
    - Researchers compete for limited time slots

    **2. Materials Discovery:**
    - High-temperature superconductors
    - Thermoelectric materials
    - Battery materials
    - Quantum materials

    **3. Broader Applications:**
    - **Pharmaceuticals**: Drug crystal structure analysis
    - **Engineering**: Stress analysis in materials
    - **Energy**: Fuel cell and battery research
    - **Quantum computing**: Material characterization

    **Your Role as Experimentalist:**
    Even with automation, humans need to:
    - Set scientific goals
    - Interpret results
    - Make strategic decisions
    - Understand the underlying physics
    """)

    st.markdown("---")
    st.info("""
    **Quick Reference:**
    - **Goal**: Maximize Fisher score within 12 hours
    - **Key temperatures**: 50K and 150K (transition points)
    - **Both samples**: Pressurized AND non-pressurized
    - **Watch time**: Each scan costs time!
    - **Use theory lines**: They guide you to the right energy range
    """)



# ============================================================
# Run the app
# ============================================================
if __name__ == "__main__":
    main()
