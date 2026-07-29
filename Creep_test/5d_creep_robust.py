import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
from numba import njit

# ============ 字体路径 ============
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# ============ 样式变量定义 ============
font_family = 'Times New Roman'
font_weight = 'normal'
math_fontset = 'stix'
math_rm = 'Times New Roman'
math_it = 'Times New Roman:italic'
math_bf = 'Times New Roman:bold'

title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 30
legend_title_fontsize = 35

axes_linewidth = 2
xtick_major_width = 2
ytick_major_width = 2
xtick_major_size = 10
ytick_major_size = 10
grid_linewidth = 1
grid_alpha = 0.4
lines_linewidth = 4
lines_markersize = 15

xtick_direction = 'in'
ytick_direction = 'in'
xtick_top = False
ytick_right = False

figure_dpi = 100
savefig_dpi = 300

# ============ 应用全局设置 ============
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.family'] = font_family

plt.rcParams.update({
    'mathtext.fontset': math_fontset,
    'mathtext.rm': math_rm,
    'mathtext.it': math_it,
    'mathtext.bf': math_bf,
    'font.weight': font_weight,
    'axes.titlesize': title_fontsize,
    'axes.labelsize': label_fontsize,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': legend_fontsize,
    'legend.title_fontsize': legend_title_fontsize,
    'axes.linewidth': axes_linewidth,
    'xtick.major.width': xtick_major_width,
    'ytick.major.width': ytick_major_width,
    'xtick.major.size': xtick_major_size,
    'ytick.major.size': ytick_major_size,
    'grid.linewidth': grid_linewidth,
    'grid.alpha': grid_alpha,
    'lines.linewidth': lines_linewidth,
    'lines.markersize': lines_markersize,
    'figure.dpi': figure_dpi,
    'savefig.dpi': savefig_dpi,
    'xtick.direction': xtick_direction,
    'ytick.direction': ytick_direction,
    'xtick.top': xtick_top,
    'ytick.right': ytick_right,
})

# ============ 保存路径 ============
save_path = "/home/tyt/project/Black_hole/Creep_test"
os.makedirs(save_path, exist_ok=True)

# ===================== 核心求解器（重构版） =====================


@njit(cache=True)
def solve_initial_lambda(sigma, tol=1e-12, max_iter=50):
    """
    Solve lambda_0 from sigma = lambda - 1/lambda^2  (neo-Hookean, Eq.8).

    Uses Newton's method with small-strain initial guess lambda_0 ~ 1 + sigma/3.
    """
    x = 1.0 + sigma / 3.0
    for _ in range(max_iter):
        x2 = x * x
        f = x - 1.0 / x2 - sigma
        if abs(f) < tol:
            break
        df = 1.0 + 2.0 / (x2 * x)
        dx = f / df
        x = x - dx
        if x < 1.0:
            x = 1.0 + 1e-12
    return x


@njit(cache=True)
def compute_taylor_coeffs(lambda0, beta=1.0):
    """
    Compute the Taylor expansion coefficients A and B.

    A from Eq.(12):  A = beta * lambda0 * (lambda0^3 - 1) / (lambda0^3 + 2)
    B from Eq.(13):  B = beta^2 * lambda0 * (lambda0^3 - 1) *
                          (lambda0^6 - 3*lambda0^4 + 10*lambda0^3 - 6*lambda0 - 2)
                          / [2 * (lambda0^3 + 2)^3]

    These give the short-time behavior: lambda(t) = lambda0 + A*t + B*t^2 + ...
    """
    l0 = lambda0
    l0_3 = l0 * l0 * l0
    denom = l0_3 + 2.0

    A = beta * l0 * (l0_3 - 1.0) / denom

    l0_4 = l0_3 * l0
    l0_6 = l0_3 * l0_3
    B_num = l0_6 - 3.0 * l0_4 + 10.0 * l0_3 - 6.0 * l0 - 2.0
    B = beta * beta * l0 * (l0_3 - 1.0) * B_num / (2.0 * denom * denom * denom)

    return A, B


@njit(cache=True)
def build_initial_guess(lambda0, A, B, sigma, beta, n_max, t_step):
    """
    Build an initial guess for the full strain history lambda(t).

    Strategy depends on stress magnitude:
    - sigma < 0.01:  pure polynomial (Eq.11), since the exponential prefactor
                     alpha = exp(-17/(1+17*sigma^2)) is negligibly small.
    - sigma >= 0.01: full SI interpolation formula (SI Eq.14).
    """
    strain = np.zeros(n_max + 1)
    strain[0] = lambda0

    if sigma < 0.01:
        # Pure polynomial initial guess: avoids exponential "pollution"
        for i in range(1, n_max + 1):
            t = i * t_step
            strain[i] = lambda0 + A * t + B * t * t
    else:
        # Full SI interpolation formula
        alpha = np.exp(-17.0 / (1.0 + 17.0 * sigma * sigma))
        c = alpha * sigma
        half_b2 = 0.5 * beta * beta
        for i in range(1, n_max + 1):
            t = i * t_step
            strain[i] = (lambda0 - c
                         + (A - beta * c) * t
                         + (B - half_b2 * c) * t * t
                         + c * np.exp(beta * t))
    return strain


@njit(cache=True)
def compute_creep_gauss_seidel(sigma, beta=1.0, t_step=0.01, n_max=4000,
                                max_iter=30, tol=1e-12, omega=0.7):
    """
    Gauss-Seidel Picard iteration for the full vitrimer creep equation.

    Key improvements over the original Jacobi-Picard approach:
    1. Gauss-Seidel:  each time step uses the most recently updated history,
       respecting the causal structure of the Volterra equation.
    2. Left-rectangular Volterra integration:  more stable than trapezoidal
       for exponential-decay kernels.
    3. Newton's method per time step with step-size control for robustness.
    4. Integral evaluated via recurrence relations for O(N) efficiency per sweep.

    Mathematical formulation
    ------------------------
    Constitutive equation (Eq.3, dimensionless, beta=1):

        sigma = e^{-t} * f(lambda(t))
                + integral_0^t  e^{-(t-tau)} * g(lambda(t), lambda(tau)) d tau

    where  f(L)     = L - 1/L^2
           g(L, L_i) = L/L_i^2 - L_i/L^2

    Left-rectangular discretization at t_n = n * dt:

        sigma = e^{-t_n} * f(lambda_n)
                + dt * sum_{i=0}^{n-1} e^{-(t_n-t_i)} * g(lambda_n, lambda_i)

    Using the decomposition  g(L, L_i) = L / L_i^2 - L_i / L^2,
    we precompute the history-dependent sums:

        S1_n = sum_{i=0}^{n-1} e^{-(t_n-t_i)} / lambda_i^2
        S2_n = sum_{i=0}^{n-1} e^{-(t_n-t_i)} * lambda_i

    Then:  F(lambda_n) = e^{-t_n}*(lambda_n - 1/lambda_n^2)
                         + dt*(lambda_n * S1_n - S2_n / lambda_n^2) - sigma = 0

    with derivative:
        F'(lambda_n) = e^{-t_n}*(1 + 2/lambda_n^3)
                       + dt*(S1_n + 2*S2_n/lambda_n^3)

    Recurrence for S1, S2:
        S1_0 = 0,  S2_0 = 0
        S1_n = e^{-dt} * S1_{n-1} + 1 / lambda_{n-1}^2
        S2_n = e^{-dt} * S2_{n-1} + lambda_{n-1}
    """
    # Step 1: initial condition
    lambda0 = solve_initial_lambda(sigma)

    # Step 2: Taylor coefficients for initial guess
    A, B = compute_taylor_coeffs(lambda0, beta)

    # Step 3: initial guess for full history
    strain = build_initial_guess(lambda0, A, B, sigma, beta, n_max, t_step)

    # Pre-allocate integral accumulators
    S1 = np.zeros(n_max + 1)
    S2 = np.zeros(n_max + 1)

    exp_dt = np.exp(-beta * t_step)
    beta_dt = beta * t_step

    # ---- Gauss-Seidel Picard iteration ----
    for k in range(max_iter):
        max_diff = 0.0
        S1[0] = 0.0
        S2[0] = 0.0

        for n in range(1, n_max + 1):
            t_n = n * t_step
            exp_tn = np.exp(-beta * t_n)

            # Recurrence: uses already-updated strain[n-1] (Gauss-Seidel)
            lam_prev = strain[n - 1]
            lam_prev_2 = lam_prev * lam_prev
            S1[n] = exp_dt * S1[n - 1] + 1.0 / lam_prev_2
            S2[n] = exp_dt * S2[n - 1] + lam_prev

            # Newton's method for lambda_n
            lam = strain[n]

            for _ in range(50):
                lam2 = lam * lam
                lam3 = lam2 * lam

                # Residual F(lambda)
                f_val = (exp_tn * (lam - 1.0 / lam2)
                         + beta_dt * (lam * S1[n] - S2[n] / lam2)
                         - sigma)

                if abs(f_val) < tol:
                    break

                # Derivative F'(lambda)
                df_val = (exp_tn * (1.0 + 2.0 / lam3)
                          + beta_dt * (S1[n] + 2.0 * S2[n] / lam3))

                dlam = f_val / df_val

                # Step-size control: limit to 50% relative change per iteration
                max_step = 0.5 * lam
                if dlam > max_step:
                    dlam = max_step
                elif dlam < -max_step:
                    dlam = -max_step

                lam_new = lam - dlam
                if lam_new < 1.0:
                    lam_new = 1.0 + 1e-12

                if abs(lam_new - lam) < tol:
                    lam = lam_new
                    break

                lam = lam_new

            # Under-relaxation for stability
            strain_new = omega * lam + (1.0 - omega) * strain[n]

            diff = abs(strain_new - strain[n])
            if diff > max_diff:
                max_diff = diff

            strain[n] = strain_new

        if max_diff < tol:
            break

    return strain


def compute_strain_rate(strain, t_step):
    """Forward difference for strain rate (avoids np.gradient oscillations)."""
    n = len(strain)
    rate = np.zeros(n)
    rate[0] = (strain[1] - strain[0]) / t_step
    for i in range(1, n):
        rate[i] = (strain[i] - strain[i - 1]) / t_step
    return rate


# ===================== 解析解参考 =====================

def analytical_small_t(sigma, beta, t_step, n_max):
    """
    Compute the analytical small-time expansion (Eq.11):
        lambda(t) = lambda0 + A*t + B*t^2

    This serves as a reference for the numerical solution at small stresses.
    """
    lambda0 = solve_initial_lambda(sigma)
    A, B = compute_taylor_coeffs(lambda0, beta)

    t = np.arange(0, n_max + 1) * t_step
    strain = lambda0 + A * t + B * t * t
    strain_rate = A + 2.0 * B * t

    return t, strain, strain_rate


# ===================== 主程序 =====================

def main():
    p = 2.0
    t_step = 0.01
    n_max = 4000
    sigma_list = [0.002, 0.02, 0.1]
    all_curves = []
    all_analytical = []

    print("=" * 60)
    print("Gauss-Seidel Picard solver for full vitrimer creep")
    print("  - Left-rectangular Volterra integration")
    print("  - Newton's method per time step")
    print("  - Stress-dependent initial guess")
    print("=" * 60)

    total_start = time.time()

    for sigma in sigma_list:
        print(f"\n--- Computing sigma = {sigma} ---")
        start_t = time.time()

        strain = compute_creep_gauss_seidel(
            sigma, beta=1.0, t_step=t_step, n_max=n_max,
            max_iter=30, tol=1e-12, omega=0.7
        )

        end_t = time.time()
        print(f"    Elapsed: {end_t - start_t:.2f} s")

        t = np.arange(0, n_max + 1) * t_step
        strain_rate = compute_strain_rate(strain, t_step)

        all_curves.append((sigma, t, strain, strain_rate))

        # Analytical small-time reference
        t_a, strain_a, rate_a = analytical_small_t(sigma, 1.0, t_step, n_max)
        all_analytical.append((sigma, t_a, strain_a, rate_a))

    total_end = time.time()
    print(f"\nAll done. Total time: {total_end - total_start:.2f} s")

    # ---- Save data ----
    df_all = pd.DataFrame()
    for sigma, t_arr, strain, strain_rate in all_curves:
        df_all[f"sigma_{sigma}_time"] = t_arr
        df_all[f"sigma_{sigma}_strain"] = strain
        df_all[f"sigma_{sigma}_strain_rate"] = strain_rate

    csv_path = os.path.join(save_path, "creep_strains_robust.csv")
    df_all.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"Data saved to {csv_path}")

    # ===================== 图1: 蠕变应变 =====================
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#000000', '#1f77b4']
    dash_styles = [(None, None), (6, 3), (1, 2)]

    for i, (sigma, t_arr, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        ax.semilogy(t_arr, strain - 1.0,
                    color=color, linewidth=lines_linewidth,
                    label=f'$\\sigma_0 = {sigma}\\,G_0$ (Numerical)')

    # Overlay analytical small-time expansions (dashed)
    for i, (sigma, t_a, strain_a, rate_a) in enumerate(all_analytical):
        color = colors[i % len(colors)]
        ax.loglog(t_a[1:], strain_a[1:] - 1.0,
                  color=color, linestyle='--', linewidth=1.5, alpha=0.6)

    ax.set_xlabel('Scaled time $\\beta t$', fontsize=label_fontsize)
    ax.set_ylabel('Tensile creep strain $\\lambda - 1$', fontsize=label_fontsize)
    ax.set_title('Creep of Full Vitrimers (Gauss–Seidel Solver)', fontsize=title_fontsize, pad=20)

    ax.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    ax.tick_params(axis='both', which='major', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, bottom=True, left=True,
                   width=xtick_major_width, length=xtick_major_size, labelsize=tick_fontsize)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction=xtick_direction,
                   top=xtick_top, right=ytick_right, bottom=True, left=True,
                   width=xtick_major_width * 0.75, length=xtick_major_size * 0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    fig_name = os.path.join(save_path, "creep_strain.png")
    plt.savefig(fig_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"Strain figure saved to {fig_name}")

    # ===================== 图2: 应变率 =====================
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.set_yscale('log')

    for i, (sigma, t_arr, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        mask = (strain_rate > 0) & (t_arr > 0)
        ax2.plot(t_arr[mask], strain_rate[mask],
                 color=color, linewidth=lines_linewidth,
                 label=f'$\\sigma_0 = {sigma}\\,G_0$ (Numerical)')

    # Overlay analytical strain rates (dashed)
    for i, (sigma, t_a, strain_a, rate_a) in enumerate(all_analytical):
        color = colors[i % len(colors)]
        mask = rate_a > 0
        ax2.plot(t_a[mask], rate_a[mask],
                 color=color, linestyle='--', linewidth=1.5, alpha=0.6)

    ax2.set_xlabel('Scaled time $\\beta t$', fontsize=label_fontsize)
    ax2.set_ylabel('Strain rate $d\\lambda/dt$', fontsize=label_fontsize)
    ax2.set_title('Creep Strain Rate of Full Vitrimers (Gauss–Seidel Solver)',
                  fontsize=title_fontsize, pad=20)

    ax2.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    ax2.tick_params(axis='both', which='major', direction=xtick_direction,
                    top=xtick_top, right=ytick_right, bottom=True, left=True,
                    width=xtick_major_width, length=xtick_major_size, labelsize=tick_fontsize)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='minor', direction=xtick_direction,
                    top=xtick_top, right=ytick_right, bottom=True, left=True,
                    width=xtick_major_width * 0.75, length=xtick_major_size * 0.5)

    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    fig_name2 = os.path.join(save_path, "creep_strain_rate.png")
    plt.savefig(fig_name2, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"Strain rate figure saved to {fig_name2}")


if __name__ == "__main__":
    main()
