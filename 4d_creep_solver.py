"""
无量纲参数空间求解蠕变行为 strain
"""


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

# ===================== 核心求解器（无量纲空间） =====================


@njit(cache=True)
def solve_initial_lambda(sigma, p, tol=1e-12, max_iter=50):
    """
    求解初始瞬时弹性响应 λ₀
    方程：σ̄ = λ₀^{p-1} - λ₀^{-p-1}
    """
    if sigma <= 0.0:
        return 1.0
    x = 1.0 + sigma / (2.0 * p)
    if x < 1.0: x = 1.0 + 1e-12
    for _ in range(max_iter):
        x_p1 = x ** (p - 1)
        x_m1 = x ** (-p - 1)
        f = x_p1 - x_m1 - sigma
        if abs(f) < tol: break
        df = (p - 1.0) * x ** (p - 2) + (p + 1.0) * x ** (-p - 2)
        x = x - f / df
        if x < 1.0: x = 1.0 + 1e-12
    return x


@njit(cache=True)
def compute_taylor_coeffs(lambda0, p, beta=1.0):
    """
    计算泰勒展开系数
    在无量纲空间中，β=1
    """
    l0 = lambda0
    pm1, pp1 = p - 1.0, p + 1.0
    f0 = l0 ** pm1 - l0 ** (-pp1)
    fp0 = pm1 * l0 ** (p - 2) + pp1 * l0 ** (-p - 2)
    fpp0 = pm1 * (p - 2) * l0 ** (p - 3) - pp1 * (p + 2) * l0 ** (-p - 3)
    A = beta * f0 / fp0
    B = (A / 2.0) * (beta - A * fpp0 / fp0 - 2.0 * p * beta / (l0 * l0 * fp0))
    return A, B


@njit(cache=True)
def build_initial_guess(lambda0, A, B, sigma, p, n_max, t_step):
    """
    构建初始猜测（无量纲空间）
    """
    strain = np.zeros(n_max + 1)
    strain[0] = lambda0
    if sigma < 0.01:
        for i in range(1, n_max + 1):
            tau = i * t_step  # 无量纲时间 τ
            strain[i] = lambda0 + A * tau + B * tau * tau
    else:
        alpha_short = np.exp(-17.0 / (1.0 + 17.0 * sigma * sigma))
        c = alpha_short * sigma
        alpha_long = 1.0 / (p - 1.0)  # β=1
        half_al2 = 0.5 * alpha_long * alpha_long
        for i in range(1, n_max + 1):
            tau = i * t_step  # 无量纲时间 τ
            strain[i] = (lambda0 - c + (A - alpha_long * c) * tau 
                         + (B - half_al2 * c) * tau * tau + c * np.exp(alpha_long * tau))
    return strain


@njit(cache=True)
def compute_creep_gauss_seidel(sigma, p=2.0, t_step=0.01, n_max=4000,
                            max_iter=30, tol=1e-12, omega=0.7):
    """
    Gauss-Seidel Picard迭代求解（无量纲空间）
    
    本构方程：σ̄ = e^{-τ} (λ^{p-1} - λ^{-p-1}) + ∫₀^τ dτ' e^{-(τ-τ')} [λ(τ)^{p-1}/λ(τ')^p - λ(τ')^p/λ(τ)^{p+1}]
    
    其中 τ = βt 是无量纲时间，β=1
    """
    lambda0 = solve_initial_lambda(sigma, p)
    A, B = compute_taylor_coeffs(lambda0, p, beta=1.0)  # β=1

    strain = build_initial_guess(lambda0, A, B, sigma, p, n_max, t_step)

    S1 = np.zeros(n_max + 1)
    S2 = np.zeros(n_max + 1)

    exp_dt = np.exp(-t_step)  # e^{-dτ}
    dtau = 1.0 - exp_dt  # dτ = 1 - e^{-dτ}

    for k in range(max_iter):
        max_diff = 0.0
        S1[0], S2[0] = 0.0, 0.0

        for n in range(1, n_max + 1):
            tau_n = n * t_step  # 无量纲时间 τ_n
            exp_tn = np.exp(-tau_n)  # e^{-τ}

            lam_prev = strain[n - 1]
            lam_prev_p = lam_prev ** p
            S1[n] = exp_dt * (S1[n - 1] + 1.0 / lam_prev_p)
            S2[n] = exp_dt * (S2[n - 1] + lam_prev_p)

            lam = strain[n]
            pm1, pp1 = p - 1.0, p + 1.0

            for _ in range(50):
                lam_pm1 = lam ** pm1
                lam_mpp1 = lam ** (-pp1)
                f_val = (exp_tn * (lam_pm1 - lam_mpp1)
                         + dtau * (lam_pm1 * S1[n] - lam_mpp1 * S2[n])
                         - sigma)
                if abs(f_val) < tol: break
                
                df_val = (exp_tn * (pm1 * lam ** (p - 2) + pp1 * lam ** (-p - 2))
                          + dtau * (pm1 * lam ** (p - 2) * S1[n]
                                    + pp1 * lam ** (-p - 2) * S2[n]))
                
                dlam = f_val / df_val
                max_step = 0.5 * lam
                if dlam > max_step: dlam = max_step
                elif dlam < -max_step: dlam = -max_step
                
                lam_new = lam - dlam
                if lam_new < 1.0: lam_new = 1.0 + 1e-12
                if abs(lam_new - lam) < tol:
                    lam = lam_new
                    break
                lam = lam_new

            strain_new = omega * lam + (1.0 - omega) * strain[n]
            diff = abs(strain_new - strain[n])
            if diff > max_diff: max_diff = diff
            strain[n] = strain_new

        if max_diff < tol: break

    return strain


def compute_strain_rate(strain, t_step):
    """前向差分计算应变率"""
    n = len(strain)
    rate = np.zeros(n)
    rate[0] = (strain[1] - strain[0]) / t_step
    for i in range(1, n):
        rate[i] = (strain[i] - strain[i - 1]) / t_step
    return rate


# ===================== 解析解参考 =====================

def analytical_small_t(sigma, p, t_step, n_max):
    """
    计算小时间解析解（无量纲空间）
    """
    lambda0 = solve_initial_lambda(sigma, p)
    A, B = compute_taylor_coeffs(lambda0, p, beta=1.0)

    tau = np.arange(0, n_max + 1) * t_step  # 无量纲时间
    strain = lambda0 + A * tau + B * tau * tau
    strain_rate = A + 2.0 * B * tau

    return tau, strain, strain_rate


# ===================== 主程序 =====================

def main():
    p = 2.0  # 本构指数
    
    # 无量纲时间步长和最大步数
    t_step = 0.01  # dτ
    n_max = 4000   # 最大无量纲时间步数
    
    # 无量纲应力列表（σ̄ = σ/μ）
    sigma_list = [0.002, 0.02, 0.1]
    
    print("=" * 60)
    print("Gauss-Seidel Picard solver for full vitrimer creep (Dimensionless Space)")
    print("  - Using dimensionless time τ = βt (β=1)")
    print("  - Constitutive equation without explicit β")
    print("  - Left-rectangular Volterra integration")
    print("  - Newton's method per time step")
    print("=" * 60)

    total_start = time.time()
    all_curves = []
    all_analytical = []

    for sigma in sigma_list:
        print(f"\n--- Computing σ̄ = {sigma} ---")
        start_t = time.time()

        # 在无量纲空间中求解
        strain = compute_creep_gauss_seidel(
            sigma, p=p, t_step=t_step, n_max=n_max,
            max_iter=30, tol=1e-12, omega=0.7
        )

        end_t = time.time()
        print(f"    Elapsed: {end_t - start_t:.2f} s")

        # 无量纲时间
        tau = np.arange(0, n_max + 1) * t_step
        # 应变率（无量纲）
        strain_rate = compute_strain_rate(strain, t_step)

        all_curves.append((sigma, tau, strain, strain_rate))

        # Analytical small-time reference
        tau_a, strain_a, rate_a = analytical_small_t(sigma, p, t_step, n_max)
        all_analytical.append((sigma, tau_a, strain_a, rate_a))

    total_end = time.time()
    print(f"\nAll done. Total time: {total_end - total_start:.2f} s")

    # ---- Save data ----
    df_all = pd.DataFrame()
    for sigma, tau_arr, strain, strain_rate in all_curves:
        df_all[f"sigma_{sigma}_tau"] = tau_arr
        df_all[f"sigma_{sigma}_strain"] = strain
        df_all[f"sigma_{sigma}_strain_rate"] = strain_rate

    csv_path = os.path.join(save_path, "creep_strains_dimensionless.csv")
    df_all.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"Data saved to {csv_path}")

    # ===================== 图1: 蠕变应变 =====================
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_yscale('log')

    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#000000', '#1f77b4']
    dash_styles = [(None, None), (6, 3), (1, 2)]

    for i, (sigma, tau_arr, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        ax.plot(tau_arr, strain - 1.0,
                    color=color, linewidth=lines_linewidth,
                    label=f'$\\bar{{\\sigma}} = {sigma}$ (Numerical)')

    # Overlay analytical small-time expansions (dashed)
    for i, (sigma, tau_a, strain_a, rate_a) in enumerate(all_analytical):
        color = colors[i % len(colors)]
        ax.plot(tau_a[1:], strain_a[1:] - 1.0,
                  color=color, linestyle='--', linewidth=1.5, alpha=0.6)

    ax.set_xlabel('Dimensionless time $\\tau = \\beta t$', fontsize=label_fontsize)
    ax.set_ylabel('Tensile creep strain $\\lambda - 1$', fontsize=label_fontsize)
    ax.set_title('Creep of Full Vitrimers (Gauss–Seidel Solver - Dimensionless Space)', fontsize=title_fontsize, pad=20)

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
    fig_name = os.path.join(save_path, "creep_strain_dimensionless.png")
    plt.savefig(fig_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"Strain figure saved to {fig_name}")

    # ===================== 图2: 应变率 =====================
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.set_yscale('log')

    for i, (sigma, tau_arr, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        mask = (strain_rate > 0) & (tau_arr > 0)
        ax2.plot(tau_arr[mask], strain_rate[mask],
                 color=color, linewidth=lines_linewidth,
                 label=f'$\\bar{{\\sigma}} = {sigma}$ (Numerical)')

    # Overlay analytical strain rates (dashed)
    for i, (sigma, tau_a, strain_a, rate_a) in enumerate(all_analytical):
        color = colors[i % len(colors)]
        mask = rate_a > 0
        ax2.plot(tau_a[mask], rate_a[mask],
                 color=color, linestyle='--', linewidth=1.5, alpha=0.6)

    ax2.set_xlabel('Dimensionless time $\\tau = \\beta t$', fontsize=label_fontsize)
    ax2.set_ylabel('Strain rate $d\\lambda/d\\tau$', fontsize=label_fontsize)
    ax2.set_title('Creep Strain Rate of Full Vitrimers (Gauss–Seidel Solver - Dimensionless Space)',
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
    fig_name2 = os.path.join(save_path, "creep_strain_rate_dimensionless.png")
    plt.savefig(fig_name2, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"Strain rate figure saved to {fig_name2}")


if __name__ == "__main__":
    main()
