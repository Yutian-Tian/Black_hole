"""
物理时间空间求解蠕变行为 (T-Correction Model)
对应本构方程：
sigma(t) = e^{-(1-a+a*lambda)t}(lambda^{p-1} - lambda^{-p-1}) + 
           \int_0^t [1-a+a*lambda(t)] dt' e^{-[1-a+a*lambda(t)](t-t')} [lambda(t)^{p-1}/lambda(t')^p - lambda(t')^p/lambda(t)^{p+1}]
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

# ===================== 核心求解器（物理时间空间） =====================

@njit(cache=True)
def solve_initial_lambda(sigma, p, tol=1e-12, max_iter=50):
    """
    求解初始瞬时弹性响应 λ₀
    方程：σ = λ₀^{p-1} - λ₀^{-p-1}
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
def compute_creep_gauss_seidel(sigma, p=2.0, a=0.5, t_step=0.01, n_max=4000,
                               max_iter=30, tol=1e-12, omega=0.7, picard_max=5):
    """
    使用 Picard + Newton 迭代求解 T-Correction 本构方程（物理时间空间 t）
    """
    lambda0 = solve_initial_lambda(sigma, p)
    strain = np.zeros(n_max + 1)
    strain[0] = lambda0

    for n in range(1, n_max + 1):
        # 初始猜测为上一时间步结果
        lam = strain[n - 1]

        # 外层 Picard 循环：迭代更新 R(t) = 1-a+a*lambda(t)
        for picard in range(picard_max):
            # 1. 根据当前的 lam 计算当前时刻的衰减率 R
            R = 1.0 - a + a * lam
            if R < 0.0:
                R = 1e-12  # 防止除以零
            
            exp_R_dt = np.exp(-R * t_step)
            weight = 1.0
            
            # 2. 利用左矩形法则进行显式积分计算历史项
            # 由于 R 依赖于当前时刻的 lambda，必须对历史重新积分
            sum_I1 = 0.0  # 积分项: \int R * e^{-R(t-t')} * lambda(t')^{-p} dt'
            sum_I2 = 0.0  # 积分项: \int R * e^{-R(t-t')} * lambda(t')^{p} dt'
            
            for k in range(n - 1, -1, -1):
                # weight 对应 e^{-R*(n-1-k)*dt}，integ_w 则对应精确积分区间权重
                integ_w = weight * (1.0 - exp_R_dt)
                lam_k = strain[k]
                sum_I1 += integ_w * (lam_k ** (-p))
                sum_I2 += integ_w * (lam_k ** p)
                weight *= exp_R_dt
            
            # weight 此时变为 e^{-R * n * dt}，即 A_coeff = e^{-R(t)*t}
            A_coeff = weight 
            
            # 3. 内层 Newton 循环：在 R(t) 固定的情况下求解 lambda(t)
            lam_new = lam
            for newton in range(max_iter):
                lam_pm1 = lam_new ** (p - 1.0)
                lam_mpp1 = lam_new ** (-p - 1.0)
                
                # f_val = e^{-R*t}(lambda^{p-1} - lambda^{-p-1}) + lambda^{p-1}I1 - lambda^{-p-1}I2 - sigma
                f_val = A_coeff * (lam_pm1 - lam_mpp1) + lam_pm1 * sum_I1 - lam_mpp1 * sum_I2 - sigma
                
                if abs(f_val) < tol:
                    break
                
                # 对 lambda 求导，此时的 A_coeff, sum_I1, sum_I2 均是定值
                df_val = (A_coeff + sum_I1) * (p - 1.0) * lam_new ** (p - 2.0) + \
                         (A_coeff + sum_I2) * (p + 1.0) * lam_new ** (-p - 2.0)
                
                lam_new = lam_new - f_val / df_val
                if lam_new < 1.0:
                    lam_new = 1.0 + 1e-12
            
            # 4. 应用超松弛因子更新
            lam = omega * lam_new + (1.0 - omega) * lam
            if abs(lam - lam_new) < tol:
                break
                
        strain[n] = lam

    return strain


def compute_strain_rate(strain, t_step):
    """前向差分计算应变率"""
    n = len(strain)
    rate = np.zeros(n)
    rate[0] = (strain[1] - strain[0]) / t_step
    for i in range(1, n):
        rate[i] = (strain[i] - strain[i - 1]) / t_step
    return rate


# ===================== 主程序 =====================

def main():
    # 参数设置
    p = 2.0              # 本构指数
    a = 0.1             # 新引入的材料参数 a (可修改)
    t_step = 0.01        # 物理时间步长
    n_max = 4000         # 最大物理时间步数
    
    # 施加应力列表
    sigma_list = [0.01, 0.02, 0.05]
    
    print("=" * 60)
    print("Gauss-Seidel Picard solver for T-Correction creep model (Physical Time Space)")
    print(f"  - Material Parameter a = {a}")
    print(f"  - Decay Rate: R(t) = 1 - {a} + {a}*lambda(t) (T-Correction)")
    print("  - Exact Volterra integration with variable decay rate (Left-Rectangular)")
    print("=" * 60)

    total_start = time.time()
    all_curves = []

    for sigma in sigma_list:
        print(f"\n--- Computing σ = {sigma} ---")
        start_t = time.time()

        # 求解 T-Correction 模型
        strain = compute_creep_gauss_seidel(
            sigma, p=p, a=a, t_step=t_step, n_max=n_max,
            max_iter=30, tol=1e-12, omega=0.7, picard_max=5
        )

        end_t = time.time()
        print(f"    Elapsed: {end_t - start_t:.2f} s")

        # 物理时间数组
        t_arr = np.arange(0, n_max + 1) * t_step
        # 应变率
        strain_rate = compute_strain_rate(strain, t_step)

        all_curves.append((sigma, t_arr, strain, strain_rate))

    total_end = time.time()
    print(f"\nAll done. Total time: {total_end - total_start:.2f} s")

    # ---- 保存数据 ----
    df_all = pd.DataFrame()
    for sigma, t_arr, strain, strain_rate in all_curves:
        df_all[f"sigma_{sigma}_time"] = t_arr
        df_all[f"sigma_{sigma}_strain"] = strain
        df_all[f"sigma_{sigma}_strain_rate"] = strain_rate

    csv_path = os.path.join(save_path, "creep_strains_TCorrection.csv")
    df_all.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"Data saved to {csv_path}")

    # ===================== 图1: 蠕变应变 =====================
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_yscale('log')

    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#000000', '#1f77b4']

    for i, (sigma, t_arr, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        ax.plot(t_arr, strain - 1.0,
                color=color, linewidth=lines_linewidth,
                label=f'$\\sigma = {sigma}$ (T-Correction Model, a={a})')

    ax.set_xlabel('Dimensionless time $\\beta t$', fontsize=label_fontsize)
    ax.set_ylabel('Tensile creep strain $\\lambda - 1$', fontsize=label_fontsize)
    ax.set_title(f'Creep of Full Vitrimers (T-Correction Model, $a={a}$)', 
                 fontsize=title_fontsize, pad=20)

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
    fig_name = os.path.join(save_path, "creep_strain_TCorrection.png")
    plt.savefig(fig_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"Strain figure saved to {fig_name}")

    # ===================== 图2: 应变率 =====================
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    for i, (sigma, t_arr, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        mask = (strain_rate > 0) & (t_arr > 0)
        ax2.plot(t_arr[mask], strain_rate[mask],
                 color=color, linewidth=lines_linewidth,
                 label=f'$\\sigma = {sigma}$ (T-Correction Model, a={a})')

    ax2.set_xlabel('Dimensionless time $\\beta t$', fontsize=label_fontsize)
    ax2.set_ylabel('Strain rate $d\\lambda/dt$', fontsize=label_fontsize)
    ax2.set_title(f'Creep Strain Rate of Full Vitrimers (T-Correction Model, $a={a}$)',
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
    fig_name2 = os.path.join(save_path, "creep_strain_rate_TCorrection.png")
    plt.savefig(fig_name2, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"Strain rate figure saved to {fig_name2}")


if __name__ == "__main__":
    main()