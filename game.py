import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider, CheckButtons
import jax
# jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from scipy.stats import poisson
import datetime
from collections import defaultdict
import pandas as pd


# ============================================================
# Set a pretty style
# ============================================================
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'whitesmoke'

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
game_over = False

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

    # --- Linear segments (anchored correctly) ---
    h1 = hwhm0 * (1.0 + m1 * (T - Tmin))
    h1_T1 = hwhm0 * (1.0 + m1 * (T1 - Tmin))

    h2 = h1_T1 + hwhm0 * m2 * (T - T1)
    h2_T2 = h1_T1 + hwhm0 * m2 * (T2 - T1)

    h3 = h2_T2 + hwhm0 * m3 * (T - T2)

    # --- Smooth weights ---
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
    
    # Handle partial scans: if npts is less than requested due to time constraints
    # we still sample evenly across the energy range
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
    global all_scans

    if len(all_scans) == 0:
        return -np.inf, -np.inf, -np.inf

    # Extract arrays
    E = jnp.array(all_scans['Energy'].values)
    T = jnp.array(all_scans['T'].values)
    pressurized = jnp.array(all_scans['pressurized'].values)
    count_time = jnp.array(all_scans['count_time'].values)

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

    # Jacobian: shape (Ndata, 6)
    J = jax.jacobian(lambda_model, argnums=0)(
        theta0, E, T, pressurized
    )

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


def compute_hwhm_err(df):
    args = jnp.array(df[['Energy', 'T', 'amp', 'E0', 'hwhm', 'bg']].values.T)
    return hwhm_subfunc(args)

@jax.jit
def hwhm_subfunc(args):
    # ------------------------------------------------------------------
    # Jacobian of λ wrt slopes
    # shape: (Ndata, 6)
    # ------------------------------------------------------------------
    J = jax.vmap(lambda *a: jax.jacobian(model, argnums=4)(*a))(*args)
    J = J.reshape(-1,1)
    lam = jax.vmap(model)(*args)

    # Guard against numerical issues
    lam = jnp.clip(lam, 1e-12, jnp.inf)
    # ------------------------------------------------------------------
    # Fisher Information: Jᵀ diag(1/λ) J
    # ------------------------------------------------------------------
    W = 1.0 / lam
    FI = J.T @ (J * W[:, None])
    FI = FI.squeeze()
    err = (1/jnp.sqrt(jnp.maximum(FI, 0)))
    return err

def warmup_jax():
    print("Warming up JAX...")

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

    # # Compile Fisher Jacobian
    # jac = jax.jit(jax.jacobian(lambda_model))
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
                               slope1np=TRUE_SLOPE1_NON_PRESSURIZED, slope2np=TRUE_SLOPE2_NON_PRESSURIZED, slope3np=TRUE_SLOPE3_NON_PRESSURIZED,
                               
    )
    print("JAX warm-up complete.")

default_count_time = 60.0
default_npts = 30
default_E_range = 6.0
default_T_center = 50.0
default_scan_center = 5.0

# ============================================================
# Data Storage and Planned Scans
# ============================================================
all_scans = pd.DataFrame({
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

# Store planned scans for Fisher computation
used_time = 0.0
show_theory = True

# ============================================================
# Enhanced UI Layout - Bottom information panel
# ============================================================
fig = plt.figure(figsize=(16, 12))
plt.subplots_adjust(left=0.05, bottom=0.2, right=0.98, top=0.95)

# Main plot areas
ax_data_non_pressurized = fig.add_subplot(2, 2, 1)
ax_data_non_pressurized.set_title("Non-Pressurized Energy Scans", fontsize=11, fontweight='bold')
ax_data_non_pressurized.set_xlabel("Energy [meV]", fontsize=9)
ax_data_non_pressurized.set_ylabel("Counts per Second", fontsize=9)
ax_data_non_pressurized.grid(True, alpha=0.3)
ax_data_non_pressurized.tick_params(labelsize=8)

ax_data_pressurized = fig.add_subplot(2, 2, 2)
ax_data_pressurized.set_title("Pressurized Energy Scans", fontsize=11, fontweight='bold')
ax_data_pressurized.set_xlabel("Energy [meV]", fontsize=9)
ax_data_pressurized.set_ylabel("Counts per Second", fontsize=9)
ax_data_pressurized.grid(True, alpha=0.3)
ax_data_pressurized.tick_params(labelsize=8)

ax_fit = fig.add_subplot(2, 2, 3)
ax_fit.set_title("Temperature-Dependent Linewidth HWHM(T)", fontsize=11, fontweight='bold')
ax_fit.set_xlabel("Temperature [K]", fontsize=9)
ax_fit.set_ylabel("HWHM [meV]", fontsize=9)
ax_fit.grid(True, alpha=0.3)
ax_fit.tick_params(labelsize=8)

ax_controls = fig.add_subplot(2, 2, 4)
ax_controls.set_title("Experiment Controls", fontsize=11, fontweight='bold')
ax_controls.axis('off')

# Colormaps
cmap_non_pressurized = plt.get_cmap('viridis')
cmap_pressurized = plt.get_cmap('plasma')

# ============================================================
# Information Display Area at Bottom
# ============================================================
from matplotlib.patches import Rectangle
info_height = 0.12
info_background = Rectangle((0.02, 0.02), 0.96, info_height,
                           transform=fig.transFigure, facecolor='whitesmoke',
                           edgecolor='gray', linewidth=1, alpha=0.9, zorder=0)
fig.patches.append(info_background)

# Time displays on left side
time_title = fig.text(0.03, 0.11, "TIME STATUS", fontsize=10, fontweight='bold',
                     color='darkblue', transform=fig.transFigure)

time_display_text = fig.text(0.03, 0.08, 
                            f"Time Used: {datetime.timedelta(seconds=int(used_time))}", 
                            fontsize=10, fontweight='bold', color='black',
                            transform=fig.transFigure)

time_budget_text = fig.text(0.03, 0.05, 
                           f"Time Budget: {datetime.timedelta(seconds=int(TIME_BUDGET))}", 
                           fontsize=10, fontweight='bold', color='black',
                           transform=fig.transFigure)

# Progress bar in the middle
progress_bar_ax = fig.add_axes([0.03, 0.025, 0.3, 0.01])
progress_bar_ax.axis('off')
progress_bar_line, = progress_bar_ax.plot([0, 1], [0.5, 0.5], linewidth=8, color='lightblue', solid_capstyle='round')
progress_bar_bg_line, = progress_bar_ax.plot([0, 1], [0.5, 0.5], linewidth=8, color='lightgray', solid_capstyle='round', alpha=0.3)

# Fisher scores on right side
fisher_title = fig.text(0.4, 0.11, "FISHER INFORMATION SCORES", fontsize=10, fontweight='bold',
                       color='purple', transform=fig.transFigure)

fisher_non_text = fig.text(0.4, 0.08,
                          "Non-Pressurized Score: 0.00", fontsize=10, fontweight='bold',
                          color='blue', transform=fig.transFigure)

fisher_press_text = fig.text(0.4, 0.05,
                           "Pressurized Score: 0.00", fontsize=10, fontweight='bold',
                           color='red', transform=fig.transFigure)

fisher_total_text = fig.text(0.6, 0.08,
                           "TOTAL SCORE: 0.00", fontsize=12, fontweight='bold',
                           color='purple', transform=fig.transFigure)

# Score explanation
score_explanation = fig.text(0.6, 0.05,
                           "Higher score = Better measurement precision | Score = -∞ when measurement is insufficient",
                           fontsize=8, color='gray', style='italic', transform=fig.transFigure)

# ============================================================
# Sliders Layout in Controls Panel
# ============================================================
controls_rect = ax_controls.get_position()
controls_left = controls_rect.x0
controls_bottom = controls_rect.y0
controls_width = controls_rect.width
controls_height = controls_rect.height

slider_width = 0.35
slider_height = 0.03
vertical_spacing = 0.04
start_y = controls_bottom + controls_height - 0.05

# Create sliders
ax_temp_slider = plt.axes([controls_left + 0.05, start_y, slider_width, slider_height])
temp_slider = Slider(ax_temp_slider, 'Temperature [K]', 2.0, 270.0, valinit=default_T_center, valstep=1.0)

ax_ct_slider = plt.axes([controls_left + 0.05, start_y - vertical_spacing, slider_width, slider_height])
ct_slider = Slider(ax_ct_slider, 'Time [s]', 1.0, 1800.0, valinit=default_count_time, valstep=1.0)

ax_np_slider = plt.axes([controls_left + 0.05, start_y - 2*vertical_spacing, slider_width, slider_height])
np_slider = Slider(ax_np_slider, '# of Points', 1, 100, valinit=default_npts, valstep=1)

ax_er_slider = plt.axes([controls_left + 0.05, start_y - 3*vertical_spacing, slider_width, slider_height])
er_slider = Slider(ax_er_slider, 'Energy Range', 1.0, 8.0, valinit=default_E_range, valstep=0.1)

ax_sc_slider = plt.axes([controls_left + 0.05, start_y - 4*vertical_spacing, slider_width, slider_height])
sc_slider = Slider(ax_sc_slider, 'Center [meV]', 0.1, 20.0, valinit=default_scan_center, valstep=0.1)

# Buttons
button_width = 0.17
button_height = 0.04
button_spacing = 0.02

ax_run_non = plt.axes([controls_left + 0.05, start_y - 5.5*vertical_spacing, button_width, button_height])
btn_run_non = Button(ax_run_non, 'Non-Pressurized', color='lightblue', hovercolor='skyblue')

ax_run_press = plt.axes([controls_left + 0.05 + button_width + button_spacing, start_y - 5.5*vertical_spacing, button_width, button_height])
btn_run_press = Button(ax_run_press, 'Pressurized', color='lightcoral', hovercolor='red')

ax_clr = plt.axes([controls_left + 0.05, start_y - 6.5*vertical_spacing, button_width, button_height])
btn_clr = Button(ax_clr, 'Clear All', color='lightgray', hovercolor='darkgray')

# Theory toggle button
ax_theory = plt.axes([controls_left + 0.05 + button_width + button_spacing, start_y - 6.5*vertical_spacing, button_width, button_height])
btn_theory = Button(ax_theory, 'Hide Theory', color='lightgreen', hovercolor='green')

# Game over text (initially hidden)
game_over_text = fig.text(0.5, 0.5, "", ha='center', va='center', 
                         fontsize=20, fontweight='bold', color='darkred', 
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
                         transform=fig.transFigure)
game_over_text.set_visible(False)

# ============================================================
# Parameter Validation
# ============================================================
def get_current_parameters():
    return {
        'T_center': temp_slider.val,
        'count_time': ct_slider.val,
        'npts': int(np_slider.val),
        'E_range': er_slider.val,
        'scan_center': sc_slider.val
    }

def format_time_delta(seconds):
    """Format seconds into flexible hh:mm:ss format that allows hours > 99"""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds_remaining = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds_remaining:02d}"

def trigger_game_over():
    """Trigger game over state"""
    global game_over
    if game_over:  # Already triggered
        return
    
    game_over = True
    
    non_score, press_score, total_score = compute_fisher_scores()
    
    # Format scores
    def format_score(score):
        if score == -np.inf:
            return "-∞"
        else:
            return f"{score:.2f}"
    
    non_score_str = format_score(non_score)
    press_score_str = format_score(press_score)
    total_score_str = format_score(total_score)
    
    game_over_text.set_text(f"GAME OVER\nTime Budget Exhausted!\n\n"
                           f"Non-Press Score: {non_score_str}\n"
                           f"Press Score: {press_score_str}\n"
                           f"Total Score: {total_score_str}\n\n"
                           f"Click 'Clear All' to reset")
    game_over_text.set_visible(True)
    
    # Disable scan buttons
    btn_run_non.color = 'gray'
    btn_run_press.color = 'gray'
    btn_run_non.hovercolor = 'gray'
    btn_run_press.hovercolor = 'gray'
    
    # Force the button to update its appearance
    btn_run_non.ax.set_facecolor('gray')
    btn_run_press.ax.set_facecolor('gray')
    
    fig.canvas.draw_idle()

def update_fisher_scores():
    """Update all Fisher score displays immediately"""
    non_score, press_score, total_score = compute_fisher_scores()
    
    # Format scores
    def format_score(score):
        if score == -np.inf:
            return "-∞"
        else:
            return f"{score:.2f}"
    
    fisher_non_text.set_text(f"Non-Pressurized Score: {format_score(non_score)}")
    fisher_press_text.set_text(f"Pressurized Score: {format_score(press_score)}")
    fisher_total_text.set_text(f"TOTAL SCORE: {format_score(total_score)}")
    
    # Color code based on score quality
    if non_score > -np.inf:
        fisher_non_text.set_color('green')
    else:
        fisher_non_text.set_color('blue')
        
    if press_score > -np.inf:
        fisher_press_text.set_color('green')
    else:
        fisher_press_text.set_color('red')
        
    if total_score > -np.inf:
        fisher_total_text.set_color('green')
    else:
        fisher_total_text.set_color('purple')
    
    fig.canvas.draw_idle()

def update_time_display():
    """Update the time display text and progress bar"""
    time_display_text.set_text(f"Time Used: {format_time_delta(used_time)}")
    time_budget_text.set_text(f"Time Budget: {format_time_delta(TIME_BUDGET)}")
    
    # Update progress bar
    time_ratio = min(used_time / TIME_BUDGET, 1.0)
    progress_bar_line.set_data([0, time_ratio], [0.5, 0.5])
    
    # Update progress bar color based on time used
    if time_ratio > 0.9:
        progress_bar_line.set_color('red')
        time_display_text.set_color('red')
    elif time_ratio > 0.7:
        progress_bar_line.set_color('orange')
        time_display_text.set_color('orange')
    else:
        progress_bar_line.set_color('lightblue')
        time_display_text.set_color('black')
    
    # update_fisher_scores()
    check_game_over()

def check_game_over():
    """Check if time budget is exceeded and show game over if needed"""
    global game_over
    if used_time >= TIME_BUDGET and not game_over:
        trigger_game_over()

# ============================================================
# Scan Management
# ============================================================
def add_scan_non_pressurized(event):
    add_scan(pressurized=False)

def add_scan_pressurized(event):
    add_scan(pressurized=True)

def add_scan(pressurized=False):
    global used_time, game_over, all_scans
    
    if game_over:
        return
    
    params = get_current_parameters()
    T = params['T_center']
    count_time = params['count_time']
    npts = params['npts']
    E_range = params['E_range']
    scan_center = params['scan_center']
    
    total_scan_time = count_time * npts
    
    # Calculate how many points we can complete with remaining time
    remaining_time = TIME_BUDGET - used_time
    
    if remaining_time <= 0:
        # No time left at all
        trigger_game_over()
        return

    used_time += total_scan_time

    # Sample actual data
    Es, counts_per_sec, errors, lam, amp, E0, hwhm, bg, counts = sample_poisson_scan(
        T, TRUE_AMP0, TRUE_HWHM0, TRUE_BG0, TRUE_E00, npts, E_range, count_time, scan_center, pressurized
    )
    
    # Check if this was the last possible scan
    if used_time >= TIME_BUDGET:
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
    all_scans = pd.concat(
        [all_scans, pd.DataFrame(rows)],
        ignore_index=True
    )
    update_time_display()
    update_fisher_scores()
    update_all_plots()

def clear_plots(event):
    """Clear all data and reset the game"""
    global used_time, game_over, all_scans
    
    # Clear all data
    all_scans = all_scans = pd.DataFrame({
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
    
    # Reset game state
    used_time = 0.0
    game_over = False

    # Hide game over text
    game_over_text.set_visible(False)

    # Re-enable scan buttons
    btn_run_non.color = 'lightblue'
    btn_run_press.color = 'lightcoral'
    btn_run_non.hovercolor = 'skyblue'
    btn_run_press.hovercolor = 'red'
    
    # Reset button face colors
    btn_run_non.ax.set_facecolor('lightblue')
    btn_run_press.ax.set_facecolor('lightcoral')
    
    # Reset plots
    ax_data_non_pressurized.clear()
    ax_data_non_pressurized.set_title("Non-Pressurized Energy Scans", fontsize=11, fontweight='bold')
    ax_data_non_pressurized.set_xlabel("Energy [meV]", fontsize=9)
    ax_data_non_pressurized.set_ylabel("Counts per Second", fontsize=9)
    ax_data_non_pressurized.grid(True, alpha=0.3)
    ax_data_non_pressurized.tick_params(labelsize=8)

    ax_data_pressurized.clear()
    ax_data_pressurized.set_title("Pressurized Energy Scans", fontsize=11, fontweight='bold')
    ax_data_pressurized.set_xlabel("Energy [meV]", fontsize=9)
    ax_data_pressurized.set_ylabel("Counts per Second", fontsize=9)
    ax_data_pressurized.grid(True, alpha=0.3)
    ax_data_pressurized.tick_params(labelsize=8)

    ax_fit.clear()
    ax_fit.set_title("Temperature-Dependent Linewidth HWHM(T)", fontsize=11, fontweight='bold')
    ax_fit.set_xlabel("Temperature [K]", fontsize=9)
    ax_fit.set_ylabel("HWHM [meV]", fontsize=9)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.tick_params(labelsize=8)

    update_time_display()
    update_fisher_scores()  # Reset Fisher scores display
    fig.canvas.draw_idle()

def toggle_theory(event):
    global show_theory
    show_theory = not show_theory
    if show_theory:
        btn_theory.label.set_text('Hide Theory')
        btn_theory.color = 'lightgreen'
    else:
        btn_theory.label.set_text('Show Theory')
        btn_theory.color = 'lightgray'
    update_all_plots()

def update_all_plots():
    """Update all data and fit plots with temperature grouping"""
    def plot_scans(ax, scans, cmap, condition_name):
        ax.clear()
        ax.set_title(f"{condition_name} Energy Scans", fontsize=11, fontweight='bold')
        ax.set_xlabel("Energy [meV]", fontsize=9)
        ax.set_ylabel("Counts per Second", fontsize=9)
        ax.grid(True, alpha=0.3)

        # temperatures = sorted(scans_by_temp.keys())
        T_min = min_t
        T_max = max_t + 20
        
        for i, (T, df) in enumerate(scans.groupby('T')):
            color_val = (T - T_min) / (T_max - T_min) if T_max > T_min else 0.5
            color = cmap(color_val)
            scans_at_T = df
            
            # for i, scan in enumerate(scans_at_T):
            label = f'T={T:.1f}K ({len(scans_at_T)} point{"s" if len(scans_at_T) > 1 else ""})'
            
            ind = np.argsort(df['Energy'].values)
            ax.errorbar(df['Energy'].values[ind], df['counts_per_sec'].values[ind], yerr=(df['error_l'], df['error_h']),
                        fmt='o', color=color, alpha=0.7, markersize=2, linestyle='none',
                        label=label, capsize=2, capthick=1, elinewidth=1)
            
            if show_theory:
                ax.plot(df['Energy'].values[ind], df['lam'].values[ind], '--', color=color, linewidth=1.5, alpha=0.8)

        # if temperatures:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8)

    # Update data plots
    plot_scans(ax_data_non_pressurized, all_scans[~all_scans['pressurized']], cmap_non_pressurized, "Non-Pressurized")
    plot_scans(ax_data_pressurized, all_scans[all_scans['pressurized']], cmap_pressurized, "Pressurized")

    # Update fit plot
    ax_fit.clear()
    ax_fit.set_title("Temperature-Dependent Linewidth HWHM(T)", fontsize=11, fontweight='bold')
    ax_fit.set_xlabel("Temperature [K]", fontsize=9)
    ax_fit.set_ylabel("HWHM [meV]", fontsize=9)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.set_ylim(0, 1)
    ax_fit.set_xlim(min_t, max_t+1)
    
    # Plot non-pressurized measurements
    df = all_scans[~all_scans['pressurized']]
    for T, _df in df.groupby('T'):
        hwhm_err = compute_hwhm_err(_df)
        ax_fit.errorbar(T, _df['hwhm'].mean(), hwhm_err,
            linestyle='none', color='blue', linewidth=2,
            marker='o', markersize=6, markeredgecolor='darkblue',
            markeredgewidth=0.5, capsize=4, capthick=2,
            label='Non-Pressurized' if T == df['T'].min() else None)
    df = all_scans[all_scans['pressurized']]
    for T, _df in df.groupby('T'):
        hwhm_err = compute_hwhm_err(_df)
        ax_fit.errorbar(T, _df['hwhm'].mean(), hwhm_err,
            linestyle='none', color='red', linewidth=2,
            marker='o', markersize=6, markeredgecolor='darkblue',
            markeredgewidth=0.5, capsize=4, capthick=2,
            label='Pressurized' if T == df['T'].min() else None)

    # # Add theoretical guides if enabled
    if show_theory:
        T_guide = np.linspace(2, 270, 100)
        hwhm_guide_non = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, False) for T_val in T_guide]
        hwhm_guide_press = [temperature_dependent_hwhm(T_val, TRUE_HWHM0, True) for T_val in T_guide]
        
        ax_fit.plot(T_guide, hwhm_guide_non, '--', color='blue', alpha=0.5, label='Non-Pressurized (expected)')
        ax_fit.plot(T_guide, hwhm_guide_press, '--', color='red', alpha=0.5, label='Pressurized (expected)')
    
    if all_scans is not None and len(all_scans) > 0:
        # Remove duplicate labels
        handles, labels = ax_fit.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_fit.legend(by_label.values(), by_label.keys(), fontsize=8)

    fig.canvas.draw_idle()

# ============================================================
# Event Connections
# ============================================================
btn_run_non.on_clicked(add_scan_non_pressurized)
btn_run_press.on_clicked(add_scan_pressurized)
btn_clr.on_clicked(clear_plots)
btn_theory.on_clicked(toggle_theory)

# Initialize displays
update_time_display()
update_fisher_scores()

# Footer
fig.text(0.5, 0.005, 
         "Optimize your measurement strategy within 12 hours! Collect scans at different temperatures to maximize Fisher information.", 
         ha='center', fontsize=9, style='italic')

warmup_jax()
plt.show()