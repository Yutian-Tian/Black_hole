import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
from numba import njit

# ============ 字体路径（如有需要可修改） ============
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
legend_fontsize = 25
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

# ===================== 2. 保存路径 =====================
save_path = "/home/tyt/project/Black_hole/5d_creep_results"
os.makedirs(save_path, exist_ok=True)

# ===================== 3. Numba 加速的核心物理函数 =====================

@njit(cache=True)
def solve_initial_lambda(sigma, p, tol=1e-10, max_iter=100):
    """求解 t=0 时刻的初始应变 λ₀（牛顿迭代法）"""
    x = 1.0
    for _ in range(max_iter):
        f = x**(p-1) - x**(-0.5*p-1) - sigma
        if abs(f) < tol:
            break
        df = (p-1) * x**(p-2) + (0.5*p+1) * x**(-0.5*p-2)
        x = x - f / df
        if x < 1e-3:
            x = 1e-3
    return x

@njit(cache=True)
def ConstitutiveEqn_val(strain_hist, t_idx, p, t_step, current_val):
    """
    计算当前时刻的无量纲应力。
    【优化点】：不再复制 strain 数组，直接传入 current_val 替代 strain[t_idx]
    """
    if t_idx == 0:
        return current_val**(p-1) - current_val**(-0.5*p-1)
    
    # 第一项：初始链的贡献（衰减项）
    term1 = np.exp(-t_idx * t_step) * (current_val**(p-1) - current_val**(-0.5*p-1))
    
    # 第二项：新形成链的积分贡献（梯形法，直接展开积分项）
    term2 = 0.0
    for i in range(t_idx):
        # A(n; i)
        exp_i = np.exp(-(t_idx - i) * t_step)
        lam_i = strain_hist[i]
        A_i = exp_i * (current_val**(p-1) / lam_i**p - lam_i**(0.5*p) / current_val**(0.5*p+1))
        
        # A(n; i+1)
        exp_i1 = np.exp(-(t_idx - (i+1)) * t_step)
        lam_i1 = strain_hist[i+1]
        A_i1 = exp_i1 * (current_val**(p-1) / lam_i1**p - lam_i1**(0.5*p) / current_val**(0.5*p+1))
        
        term2 += 0.5 * (A_i + A_i1) * t_step
        
    return term1 + term2

@njit(cache=True)
def solve_current_step(strain_hist, n, p, t_step, sigma, tol=1e-8, max_iter_bisect=100):
    """
    利用二分法求解当前时间步 n 的应变 λ
    积分时使用历史数组 strain_hist（来自上一次 Picard 全局迭代）
    """
    x0 = strain_hist[n-1]  # 从上一时刻开始找根
    
    # 寻找二分法的括号区间 [a, b] 包含零点
    a = x0
    b = x0 * 2.0
    fa = ConstitutiveEqn_val(strain_hist, n, p, t_step, a) - sigma
    fb = ConstitutiveEqn_val(strain_hist, n, p, t_step, b) - sigma
    
    # 如果同号，扩大 b 的范围，直到异号
    while fa * fb > 0 and b < 1e6:
        b *= 2.0
        fb = ConstitutiveEqn_val(strain_hist, n, p, t_step, b) - sigma
    
    # 如果扩大后依然没有异号，退回到最宽的安全区间
    if fa * fb > 0:
        a = x0
        b = 1e6
        fa = ConstitutiveEqn_val(strain_hist, n, p, t_step, a) - sigma
        fb = ConstitutiveEqn_val(strain_hist, n, p, t_step, b) - sigma
        if fa * fb > 0:
            return b  # 理论上不会发生，返回上限作为兜底
    
    # 二分法主循环
    for _ in range(max_iter_bisect):
        c = (a + b) / 2.0
        fc = ConstitutiveEqn_val(strain_hist, n, p, t_step, c) - sigma
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
    return (a + b) / 2.0

@njit(cache=True)
def compute_creep_picard_numba(sigma, p, t_step, n_max, max_iter=40, tol=1e-8):
    """使用 Picard 全局迭代法计算整条蠕变曲线"""
    # 初始化应变曲线
    strain = np.ones(n_max + 1)
    strain[0] = solve_initial_lambda(sigma, p)
    # 给初始曲线一个微弱增长的趋势（帮助更快跳出亚稳态）
    for i in range(1, n_max + 1):
        strain[i] = strain[0] * np.exp(i * t_step * 0.01)
    
    for k in range(max_iter):
        new_strain = np.zeros(n_max + 1)
        new_strain[0] = strain[0]
        
        # 逐时间步求解
        for n in range(1, n_max + 1):
            # 注意：这里传递的历史数组是上一次全局迭代的 `strain`
            # 让当前应变和历史应变在全局迭代中同时演化
            new_strain[n] = solve_current_step(strain, n, p, t_step, sigma)
        
        # 计算收敛误差
        diff = 0.0
        for i in range(n_max + 1):
            d = new_strain[i] - strain[i]
            if d < 0: 
                d = -d
            if d > diff:
                diff = d
        
        strain = new_strain
        if diff < tol:
            break
            
    return strain

# ===================== 4. 主程序（多应力计算 & 绘图） =====================
def main():
    p = 2.0
    t_step = 0.01
    n_max = 1600
    sigma_list = [0.2, 0.4, 0.6, 1.0, 3.0]
    all_curves = []

    print("开始使用 Numba 加速的 Picard 全局迭代法计算...")
    total_start = time.time()
    
    for sigma in sigma_list:
        print(f"\n--- 计算 σ = {sigma} ---")
        start_t = time.time()
        strain = compute_creep_picard_numba(sigma, p, t_step, n_max)
        end_t = time.time()
        print(f"   耗时: {end_t - start_t:.2f} 秒")
        
        dimensionless_time = np.arange(0, n_max + 1) * t_step
        all_curves.append((sigma, dimensionless_time, strain))

    total_end = time.time()
    print(f"\n✅ 全部计算完成！总耗时: {total_end - total_start:.2f} 秒")

    # 保存数据
    df_all = pd.DataFrame()
    for sigma, t, strain in all_curves:
        df_all[f"sigma_{sigma}_time"] = t
        df_all[f"sigma_{sigma}_strain"] = strain
    csv_path = os.path.join(save_path, "creep_strains_numba_picard.csv")
    df_all.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✅ 数据已保存至 {csv_path}")

    # ===================== 5. 绘图（文献图4a复现） =====================
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#000000', '#1f77b4']

    for i, (sigma, t, strain) in enumerate(all_curves):
        # 文献图4(a) 纵轴为拉伸应变 (λ - 1)，取半对数
        ax.semilogy(t, strain - 1.0,
                    color=colors[i % len(colors)],
                    linewidth=lines_linewidth,
                    label=f'$\\sigma_0 = {sigma} G_0$')

    ax.set_xlabel(f'Scaled time $\\beta t$', fontsize=label_fontsize)
    ax.set_ylabel(f'Tensile creep strain $(\lambda - 1)$', fontsize=label_fontsize)
    ax.set_title(f'Creep of Vitrimers', fontsize=title_fontsize, pad=20)

    ax.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)

    # 刻度样式
    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width,
                   length=xtick_major_size, labelsize=tick_fontsize)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width * 0.75,
                   length=xtick_major_size * 0.5)

    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.01, 1.2e5)  # 调整为 λ-1 的量级

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()

    fig_name = os.path.join(save_path, "creep_visualization_numba.png")
    plt.savefig(fig_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 图片已保存至 {fig_name}")

if __name__ == "__main__":
    main()