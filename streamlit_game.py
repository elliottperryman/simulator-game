import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.stats import poisson
import datetime
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# ============================================================
# Streamlit Configuration
# ============================================================
st.set_page_config(layout="wide", page_title="Neutron Scattering Experiment Simulator")

# ============================================================
# Custom CSS for better styling
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .score-box {
        background-color: #EFF6FF;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #93C5FD;
        margin: 0.5rem 0;
    }
    .time-box {
        background-color: #F0F9FF;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #7DD3FC;
    }
    .stButton button {
        width: 100%;
        font-weight: bold;
    }
    .stSlider {
        margin-bottom: 1rem;
    }
    .progress-container {
        background-color: #E5E7EB;
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
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
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'used_time' not in st.session_state:
    st.session_state.used_time = 0.0
if 'all_scans' not in st.session_state:
    st.session_state.all_scans = pd.DataFrame({
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
if 'show_theory' not in st.session_state:
    st.session_state.show_theory = True
if 'temp_slider' not in st.session_state:
    st.session_state.temp_slider = 50.0
if 'ct_slider' not in st.session_state:
    st.session_state.ct_slider = 60.0
if 'np_slider' not in st.session_state:
    st.session_state.np_slider = 30
if 'er_slider' not in st.session_state:
    st.session_state.er_slider = 6.0
if 'sc_slider' not in st.session_state:
    st.session_state.sc_slider = 5.0

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
    J = jax.jacobian(lambda_model, argnums=0)(theta0, E, T, pressurized)

    # Fisher matrix
    W = count_time / lam
    FI = J.T @ (J * W[:, None])

    # Separate blocks
    FI_press = FI[:3, :3]
    FI_non   = FI[3:, 3:]

    def logdet(mat):
        sign, ld = jnp.linalg.slogdet(mat)
        return jnp.where(sign > 0, ld, -jnp.inf)

    lp = logdet(FI_press)
    lnp = logdet(FI_non)

    return float(lnp), float(lp), float(lnp + lp)

@jax.jit
def hwhm_subfunc(args):
    J = jax.vmap(lambda *a: jax.jacobian(model, argnums=4)(*a))(*args)
    J = J.reshape(-1,1)
    lam = jax.vmap(model)(*args)
    lam = jnp.clip(lam, 1e-12, jnp.inf)
    
    W = 1.0 / lam
    FI = J.T @ (J * W[:, None])
    FI = FI.squeeze()
    err = (1/jnp.sqrt(jnp.maximum(FI, 0)))
    return err

def compute_hwhm_err(df):
    args = jnp.array(df[['Energy', 'T', 'amp', 'E0', 'hwhm', 'bg']].values.T)
    return hwhm_subfunc(args)

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
    if st.session_state.used_time + total_scan_time > TIME_BUDGET:
        # Calculate how many points we can actually measure
        remaining_time = TIME_BUDGET - st.session_state.used_time
        if remaining_time <= 0:
            trigger_game_over()
            return
            
        npts_possible = int(remaining_time / count_time)
        if npts_possible == 0:
            trigger_game_over()
            return
            
        # Use the smaller number of points
        npts = npts_possible
        total_scan_time = count_time * npts
    
    st.session_state.used_time += total_scan_time

    # Sample actual data
    Es, counts_per_sec, errors, lam, amp, E0, hwhm, bg, counts = sample_poisson_scan(
        T, TRUE_AMP0, TRUE_HWHM0, TRUE_BG0, TRUE_E00, npts, E_range, count_time, scan_center, pressurized
    )
    
    if st.session_state.used_time >= TIME_BUDGET:
        trigger_game_over()
    
    rows = []
    for E, cps, e, l, c in zip(Es, counts_per_sec, errors.T, lam, counts):
        rows.append({
            'T': T,
            'Energy': E,
            'counts_per_sec': cps,
            'error_l': e[0],
            'error_h': e[1],
            'lam': l,
            'amp': amp,
            'E0': E0,
            'hwhm': float(hwhm),
            'bg': bg,
            'counts': c,
            'count_time': count_time,
            'pressurized': pressurized,
        })
    
    st.session_state.all_scans = pd.concat(
        [st.session_state.all_scans, pd.DataFrame(rows)],
        ignore_index=True
    )
    st.rerun()

def clear_plots():
    """Clear all data and reset the game"""
    st.session_state.all_scans = pd.DataFrame({
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
    plt.style.use('seaborn-v0_8-darkgrid')
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['axes.facecolor'] = 'whitesmoke'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Neutron Scattering Experiment Simulator', fontsize=16, fontweight='bold')
    
    # Colormaps
    cmap_non_pressurized = plt.get_cmap('viridis')
    cmap_pressurized = plt.get_cmap('plasma')
    
    # Plot non-pressurized data
    ax1.set_title("Non-Pressurized Energy Scans", fontsize=11, fontweight='bold')
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
    ax2.set_title("Pressurized Energy Scans", fontsize=11, fontweight='bold')
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
    ax3.set_xlim(min_t, max_t + 1)
    
    if len(st.session_state.all_scans) > 0:
        # Non-pressurized measurements
        non_press_scans = st.session_state.all_scans[~st.session_state.all_scans['pressurized']]
        for T, df in non_press_scans.groupby('T'):
            if len(df) > 0:
                hwhm_err = compute_hwhm_err(df)
                ax3.errorbar(T, df['hwhm'].mean(), hwhm_err,
                            linestyle='none', color='blue', linewidth=2,
                            marker='o', markersize=6, markeredgecolor='darkblue',
                            markeredgewidth=0.5, capsize=4, capthick=2,
                            label='Non-Pressurized' if T == non_press_scans['T'].min() else None)
        
        # Pressurized measurements
        press_scans = st.session_state.all_scans[st.session_state.all_scans['pressurized']]
        for T, df in press_scans.groupby('T'):
            if len(df) > 0:
                hwhm_err = compute_hwhm_err(df)
                ax3.errorbar(T, df['hwhm'].mean(), hwhm_err,
                            linestyle='none', color='red', linewidth=2,
                            marker='o', markersize=6, markeredgecolor='darkred',
                            markeredgewidth=0.5, capsize=4, capthick=2,
                            label='Pressurized' if T == press_scans['T'].min() else None)
        
        # Theoretical guides
        if st.session_state.show_theory:
            T_guide = np.linspace(2, 270, 100)
            hwhm_guide_non = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, False) for T_val in T_guide]
            hwhm_guide_press = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, True) for T_val in T_guide]
            
            ax3.plot(T_guide, hwhm_guide_non, '--', color='blue', alpha=0.5, label='Non-Pressurized (expected)')
            ax3.plot(T_guide, hwhm_guide_press, '--', color='red', alpha=0.5, label='Pressurized (expected)')
        
        # Legend
        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax3.legend(by_label.values(), by_label.keys(), fontsize=8)
    
    # Empty axis for controls info
    ax4.axis('off')
    scan_info = f"""
    Scans collected: {len(st.session_state.all_scans)}
    Unique temperatures: {st.session_state.all_scans['T'].nunique() if len(st.session_state.all_scans) > 0 else 0}
    Non-pressurized: {len(st.session_state.all_scans[~st.session_state.all_scans['pressurized']])}
    Pressurized: {len(st.session_state.all_scans[st.session_state.all_scans['pressurized']])}
    """
    
    ax4.text(0.5, 0.5, scan_info,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax4.transAxes,
             fontsize=10,
             bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
    
    plt.tight_layout()
    return fig

# ============================================================
# Main Streamlit App
# ============================================================
st.markdown('<h1 class="main-header">Neutron Scattering Experiment Simulator</h1>', unsafe_allow_html=True)

# Header info
st.markdown("""
<div class="info-box">
    <strong>Goal:</strong> Optimize your measurement strategy within 12 hours! 
    Collect scans at different temperatures to maximize Fisher information. 
    The Fisher score quantifies how well your measurements constrain the temperature-dependent linewidth parameters.
</div>
""", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("## 🎮 Experiment Controls")
    
    # Game status
    time_ratio = st.session_state.used_time / TIME_BUDGET
    
    st.markdown(f"""
    <div class="time-box">
        <strong>Time Used:</strong> {format_time_delta(st.session_state.used_time)}<br>
        <strong>Time Budget:</strong> {format_time_delta(TIME_BUDGET)}<br>
        <strong>Remaining:</strong> {format_time_delta(max(0, TIME_BUDGET - st.session_state.used_time))}<br>
        <strong>Progress:</strong> {min(100, time_ratio*100):.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    # Custom progress bar
    progress_html, display_ratio = create_custom_progress_bar(time_ratio)
    st.markdown(progress_html, unsafe_allow_html=True)
    
    if st.session_state.game_over:
        st.error("⏰ Game Over! Time budget exhausted!")
        if st.button("🔄 Restart Game"):
            clear_plots()
    
    st.markdown("---")
    st.markdown("### Scan Parameters")
    
    # Sliders
    st.session_state.temp_slider = st.slider(
        "Temperature [K]",
        min_value=2.0,
        max_value=270.0,
        value=st.session_state.temp_slider,
        step=1.0,
        help="Center temperature for the scan"
    )
    
    st.session_state.ct_slider = st.slider(
        "Counting Time per Point [s]",
        min_value=1.0,
        max_value=1800.0,
        value=st.session_state.ct_slider,
        step=1.0,
        help="Time spent measuring each energy point"
    )
    
    st.session_state.np_slider = st.slider(
        "Number of Points",
        min_value=1,
        max_value=100,
        value=st.session_state.np_slider,
        step=1,
        help="Number of energy points in the scan"
    )
    
    st.session_state.er_slider = st.slider(
        "Energy Range [meV]",
        min_value=1.0,
        max_value=8.0,
        value=st.session_state.er_slider,
        step=0.1,
        help="Total energy range centered on scan center"
    )
    
    st.session_state.sc_slider = st.slider(
        "Energy Center [meV]",
        min_value=0.1,
        max_value=20.0,
        value=st.session_state.sc_slider,
        step=0.1,
        help="Center energy for the scan"
    )
    
    # Calculate total scan time
    total_scan_time = st.session_state.ct_slider * st.session_state.np_slider
    remaining_time = TIME_BUDGET - st.session_state.used_time
    
    st.info(f"**This scan will take:** {format_time_delta(total_scan_time)}")
    
    if total_scan_time > remaining_time and not st.session_state.game_over:
        st.warning(f"⚠️ Not enough time for full scan! You only have {format_time_delta(remaining_time)} remaining.")
    
    st.markdown("---")
    st.markdown("### Run Scans")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Non-Pressurized", 
                    disabled=st.session_state.game_over or remaining_time <= 0,
                    help="Run a scan at non-pressurized conditions",
                    type="secondary"):
            add_scan(pressurized=False)
    
    with col2:
        if st.button("🔥 Pressurized", 
                    disabled=st.session_state.game_over or remaining_time <= 0,
                    help="Run a scan at pressurized conditions",
                    type="secondary"):
            add_scan(pressurized=True)
    
    st.markdown("---")
    st.markdown("### Display Options")
    
    if st.button("🔄 Toggle Theory Lines", 
                help="Show/hide theoretical prediction lines"):
        st.session_state.show_theory = not st.session_state.show_theory
        st.rerun()
    
    if st.button("🗑️ Clear All Data", 
                help="Reset all scans and start over",
                type="primary"):
        clear_plots()
    
    st.markdown("---")
    
    # Fisher scores
    non_score, press_score, total_score = compute_fisher_scores()
    
    def format_score(score):
        if score == -np.inf:
            return "-∞"
        else:
            return f"{score:.2f}"
    
    st.markdown("### 📊 Fisher Information Scores")
    st.markdown(f"""
    <div class="score-box">
        <strong>Non-Pressurized Score:</strong> {format_score(non_score)}
    </div>
    <div class="score-box">
        <strong>Pressurized Score:</strong> {format_score(press_score)}
    </div>
    <div class="score-box" style="background-color: #E0E7FF; border-color: #4F46E5;">
        <strong>Total Score:</strong> {format_score(total_score)}
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Higher score = Better measurement precision")
    st.caption("Score = -∞ when measurement is insufficient")

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    # Create and display plots
    fig = create_plots()
    st.pyplot(fig)
    
    # Data summary
    st.markdown("### 📈 Data Summary")
    if len(st.session_state.all_scans) > 0:
        summary_data = {
            "Total Scans": len(st.session_state.all_scans),
            "Non-Pressurized Scans": len(st.session_state.all_scans[~st.session_state.all_scans['pressurized']]),
            "Pressurized Scans": len(st.session_state.all_scans[st.session_state.all_scans['pressurized']]),
            "Unique Temperatures": st.session_state.all_scans['T'].nunique(),
            "Total Measurement Time": f"{st.session_state.used_time/3600:.2f} hours",
            "Remaining Time": f"{max(0, (TIME_BUDGET - st.session_state.used_time)/3600):.2f} hours",
            "Time Used Percentage": f"{min(100, (st.session_state.used_time/TIME_BUDGET)*100):.1f}%"
        }
        
        cols = st.columns(3)
        for idx, (key, value) in enumerate(summary_data.items()):
            cols[idx % 3].metric(key, value)
    else:
        st.info("No scans collected yet. Use the controls in the sidebar to run your first scan!")

with col2:
    st.markdown("### 📝 Recent Scans")
    if len(st.session_state.all_scans) > 0:
        # Show recent scans
        recent_scans = st.session_state.all_scans.tail(5)
        for _, scan in recent_scans.iterrows():
            condition = "Pressurized" if scan['pressurized'] else "Non-Pressurized"
            st.write(f"**{condition}** at T={scan['T']:.1f}K")
            st.caption(f"Energy: {scan['Energy']:.2f} meV, Counts: {scan['counts']}")
    else:
        st.write("No scans yet")
    
    st.markdown("---")
    st.markdown("### ℹ️ How to Play")
    st.info("""
    1. Adjust scan parameters in the sidebar
    2. Run scans at different temperatures
    3. Balance between pressurized and non-pressurized conditions
    4. Maximize your Fisher score within 12 hours
    5. Try to cover a wide temperature range!
    """)

# Footer
st.markdown("---")
st.caption("Neutron Scattering Experiment Simulator | Optimize your measurement strategy within 12 hours!")