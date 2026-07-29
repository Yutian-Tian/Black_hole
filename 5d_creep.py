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
    if t_idx == 0:
        return current_val**(p-1) - current_val**(-0.5*p-1)
    
    term1 = np.exp(-t_idx * t_step) * (current_val**(p-1) - current_val**(-0.5*p-1))
    
    term2 = 0.0
    for i in range(t_idx):
        exp_i = np.exp(-(t_idx - i) * t_step)
        lam_i = strain_hist[i]
        A_i = exp_i * (current_val**(p-1) / lam_i**p - lam_i**(0.5*p) / current_val**(0.5*p+1))
        
        exp_i1 = np.exp(-(t_idx - (i+1)) * t_step)
        lam_i1 = strain_hist[i+1]
        A_i1 = exp_i1 * (current_val**(p-1) / lam_i1**p - lam_i1**(0.5*p) / current_val**(0.5*p+1))
        
        term2 += 0.5 * (A_i + A_i1) * t_step
        
    return term1 + term2

@njit(cache=True)
def solve_current_step(strain_hist, n, p, t_step, sigma, tol=1e-12, max_iter_bisect=100):
    x0 = strain_hist[n-1]
    a = x0
    b = x0 * 2.0
    fa = ConstitutiveEqn_val(strain_hist, n, p, t_step, a) - sigma
    fb = ConstitutiveEqn_val(strain_hist, n, p, t_step, b) - sigma
    
    max_extend = 50
    extend_count = 0
    
    while fa * fb > 0 and extend_count < max_extend:
        a_candidate = a * 0.5
        if a_candidate < 1e-10:
            a_candidate = 1e-10
        fa_new = ConstitutiveEqn_val(strain_hist, n, p, t_step, a_candidate) - sigma
        
        b_candidate = b * 2.0
        if b_candidate > 1e30:
            b_candidate = 1e30
        fb_new = ConstitutiveEqn_val(strain_hist, n, p, t_step, b_candidate) - sigma
        
        if fa * fb_new < 0:
            b = b_candidate
            fb = fb_new
            break
        if fa_new * fb < 0:
            a = a_candidate
            fa = fa_new
            break
        if fa_new * fb_new < 0:
            a = a_candidate
            fa = fa_new
            b = b_candidate
            fb = fb_new
            break
        
        a = a_candidate
        fa = fa_new
        b = b_candidate
        fb = fb_new
        extend_count += 1
        
        if a <= 1e-10 and b >= 1e30:
            break
    
    if fa * fb > 0:
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    
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
def compute_creep_picard_numba(sigma, p, t_step, n_max, max_iter=40, tol=1e-12, omega=0.9):
    """
    修改点：使用文献补充材料中的解析插值公式（Eq.14及其SI修正）给定高质量初始猜测
    """
    strain = np.zeros(n_max + 1)
    strain[0] = solve_initial_lambda(sigma, p)
    lambda0 = strain[0]
    beta = 1.0    # 无量纲化后的键交换率

    # ============ 根据文献公式(12), (13)计算泰勒展开系数 A 和 B ============
    l0 = lambda0
    l0_3 = l0**3
    l0_4 = l0**4
    l0_6 = l0_3**2
    
    # 公式 (12): A
    A = beta * l0 * (l0_3 - 1.0) / (l0_3 + 2.0)
    # 公式 (13): B
    B = beta**2 * l0 * (l0_3 - 1.0) * (l0_6 - 3.0*l0_4 + 10.0*l0_3 - 6.0*l0 - 2.0) / (2.0 * (l0_3 + 2.0)**3)
    
    # ============ 计算 SI 插值公式中的系数 ============
    # 预因子 alpha (极小应力下负责抑制指数项的增长)
    alpha = np.exp(-17.0 / (1.0 + 17.0 * sigma**2))
    # 常数项系数
    c = alpha * sigma 
    
    # ============ 使用插值公式构造高质量初始猜测 ============
    # 公式：lambda(t) = lambda0 - c + (A - beta*c)*t + (B - 0.5*beta^2*c)*t^2 + c*Exp[beta*t]
    for i in range(1, n_max + 1):
        t = i * t_step
        strain[i] = lambda0 - c + (A - beta*c)*t + (B - 0.5*beta**2*c)*t**2 + c * np.exp(beta*t)

    # ============ 迭代求解 ============
    for k in range(max_iter):
        new_strain = np.zeros(n_max + 1)
        new_strain[0] = strain[0]
        
        for n in range(1, n_max + 1):
            new_strain[n] = solve_current_step(strain, n, p, t_step, sigma)
        
        for i in range(n_max + 1):
            new_strain[i] = omega * new_strain[i] + (1.0 - omega) * strain[i]
        
        diff = 0.0
        for i in range(n_max + 1):
            d = abs(new_strain[i] - strain[i])
            if d > diff:
                diff = d
        
        strain = new_strain
        if diff < tol:
            break
            
    return strain

# ===================== 4. 主程序 =====================
def main():
    p = 2.0
    t_step = 0.005
    n_max = 16000
    sigma_list = [0.002]
    all_curves = []

    print("开始使用 Numba 加速的阻尼 Picard 迭代法计算... (初始解已替换为文献解析插值)")
    total_start = time.time()
    
    for sigma in sigma_list:
        print(f"\n--- 计算 σ = {sigma} ---")
        start_t = time.time()
        strain = compute_creep_picard_numba(sigma, p, t_step, n_max, omega=0.7)
        end_t = time.time()
        print(f"   耗时: {end_t - start_t:.2f} 秒")
        
        dimensionless_time = np.arange(0, n_max + 1) * t_step
        strain_rate = np.gradient(strain, t_step)
        
        all_curves.append((sigma, dimensionless_time, strain, strain_rate))

    total_end = time.time()
    print(f"\n✅ 全部计算完成！总耗时: {total_end - total_start:.2f} 秒")

    # 保存数据
    df_all = pd.DataFrame()
    for sigma, t, strain, strain_rate in all_curves:
        df_all[f"sigma_{sigma}_time"] = t
        df_all[f"sigma_{sigma}_strain"] = strain
        df_all[f"sigma_{sigma}_strain_rate"] = strain_rate
        
    csv_path = os.path.join(save_path, "creep_strains_numba_picard.csv")
    df_all.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"✅ 数据已保存至 {csv_path}")

    # ===================== 6. 绘图 =====================
    # 绘制图1：蠕变主图 (拉伸应变)
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#000000', '#1f77b4']

    for i, (sigma, t, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        ax.semilogy(t, strain - 1.0,
                    color=color,
                    linewidth=lines_linewidth,
                    label=f'\($\\sigma_0 = {sigma} G_0\$) (Numerical)')

    ax.set_xlabel(f'Scaled time $\\beta t$', fontsize=label_fontsize)
    ax.set_ylabel(f'Tensile creep strain $(\\lambda - 1)$', fontsize=label_fontsize)
    ax.set_title(f'Creep of Vitrimers', fontsize=title_fontsize, pad=20)

    ax.legend(fontsize=legend_fontsize, loc='upper left', framealpha=0.9, edgecolor='none')
    ax.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    ax.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right, bottom=True, left=True, width=xtick_major_width, length=xtick_major_size, labelsize=tick_fontsize)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor', direction=xtick_direction, top=xtick_top, right=ytick_right, bottom=True, left=True, width=xtick_major_width * 0.75, length=xtick_major_size * 0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    fig_name = os.path.join(save_path, "creep_visualization_numba.png")
    plt.savefig(fig_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 图片已保存至 {fig_name}")

    # 绘制图2：应变率图
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.set_yscale('log')

    for i, (sigma, t, strain, strain_rate) in enumerate(all_curves):
        color = colors[i % len(colors)]
        mask = (strain_rate > 0) & (t > 0)
        ax2.plot(t[mask], strain_rate[mask],
                 color=color,
                 linewidth=lines_linewidth,
                 label=f'$\\sigma_0 = {sigma} G_0$ (Numerical)')

    ax2.set_xlabel(f'Scaled time $\\beta t$', fontsize=label_fontsize)
    ax2.set_ylabel(f'Strain rate $d\\lambda/dt$', fontsize=label_fontsize)
    ax2.set_title(f'Creep strain rate of Vitrimers', fontsize=title_fontsize, pad=20)

    ax2.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    ax2.tick_params(axis='both', which='major', direction=xtick_direction, top=xtick_top, right=ytick_right, bottom=True, left=True, width=xtick_major_width, length=xtick_major_size, labelsize=tick_fontsize)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='minor', direction=xtick_direction, top=xtick_top, right=ytick_right, bottom=True, left=True, width=xtick_major_width * 0.75, length=xtick_major_size * 0.5)

    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()
    fig_name2 = os.path.join(save_path, "creep_strain_rate_numba.png")
    plt.savefig(fig_name2, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变率图已保存至 {fig_name2}")

if __name__ == "__main__":
    main()