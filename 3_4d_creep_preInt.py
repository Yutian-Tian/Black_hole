"""
预积分加速的蠕变模型（三组实验数据对比）
读取包含时间、三组应力和应变率的文件，使用固定参数计算模型，并可视化对比。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from numba import njit
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 全局绘图风格 ====================
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

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

# 结果保存路径
save_path = "/home/tyt/project/Black_hole/4d_creep_results"
os.makedirs(save_path, exist_ok=True)

# ==================== 2. 数据加载函数（三组） ====================
def load_three_groups(filepath):
    """
    读取包含三组数据（时间、应力、应变率）的 Excel/CSV 文件。
    假设文件格式：第一列为时间，后续每两列为一组（应力、应变率），共三组。
    返回：时间数组, 应力列表 [array1, array2, array3], 应变率列表 [array1, array2, array3]
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    ncols = df.shape[1]
    # 假设第一列为时间，后面6列依次为三组(应变率，应力)
    if ncols == 7:
        time = df.iloc[:, 0].values
        strain_rates = [df.iloc[:, 1].values, df.iloc[:, 3].values, df.iloc[:, 5].values]
        stresses = [df.iloc[:, 2].values, df.iloc[:, 4].values, df.iloc[:, 6].values]
    elif ncols == 6:  # 没有单独的时间列，时间由代码生成
        time = np.linspace(0, 10, df.shape[0])
        strain_rates = [df.iloc[:, 0].values, df.iloc[:, 2].values, df.iloc[:, 4].values]
        stresses = [df.iloc[:, 1].values, df.iloc[:, 3].values, df.iloc[:, 5].values]
    else:
        raise ValueError(f"数据列数不符合预期（6或7列），当前为 {ncols} 列。")

    # 简单校验：所有应变率数组长度相等
    n = len(strain_rates[0])
    for arr in strain_rates[1:] + stresses:
        if len(arr) != n:
            raise ValueError("各组数据长度不一致，请检查文件格式。")
    if len(time) != n:
        raise ValueError("时间列长度与数据不匹配。")

    return time, stresses, strain_rates

# ==================== 3. 核心物理函数（预积分加速） ====================
@njit(cache=True)
def solve_initial_lambda(sigma, p, tol=1e-10, max_iter=100):
    """使用二分法稳健求解 λ₀"""
    if sigma <= 0:
        return 1.0
    a, b = 0.5, 5.0
    fa = a**(p-1) - a**(-p-1) - sigma
    fb = b**(p-1) - b**(-p-1) - sigma
    while fa * fb > 0:
        if b > 100:
            break
        a *= 0.5
        b *= 2.0
        fa = a**(p-1) - a**(-p-1) - sigma
        fb = b**(p-1) - b**(-p-1) - sigma
    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = c**(p-1) - c**(-p-1) - sigma
        if abs(fc) < tol or (b-a)/2.0 < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2.0

@njit(cache=True)
def ConstitutiveEqn_val_fast(lam_current, p, n, t_step, I1_n, I2_n):
    """使用预积分 I1[n], I2[n] 的本构方程"""
    if lam_current < 1e-6:
        lam_current = 1e-6
    if n == 0:
        return lam_current**(p-1) - lam_current**(-p-1)
    term1 = np.exp(-n * t_step) * (lam_current**(p-1) - lam_current**(-p-1))
    term2 = lam_current**(p-1) * I1_n - lam_current**(-p-1) * I2_n
    return term1 + term2

@njit(cache=True)
def precompute_integrals(strain_hist, p, t_step, n_max):
    """预计算每个时间步的 I1 和 I2（梯形法则）"""
    I1 = np.zeros(n_max + 1)
    I2 = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            exp_i = np.exp(-(n - i) * t_step)
            exp_i1 = np.exp(-(n - (i+1)) * t_step)
            lam_inv_i = strain_hist[i] ** (-p)
            lam_inv_i1 = strain_hist[i+1] ** (-p)
            lam_pow_i = strain_hist[i] ** p
            lam_pow_i1 = strain_hist[i+1] ** p
            s1 += 0.5 * (exp_i * lam_inv_i + exp_i1 * lam_inv_i1) * t_step
            s2 += 0.5 * (exp_i * lam_pow_i + exp_i1 * lam_pow_i1) * t_step
        I1[n] = s1
        I2[n] = s2
    return I1, I2

@njit(cache=True)
def solve_current_step_fast(strain_hist, n, p, t_step, sigma, I1, I2,
                            tol=1e-8, max_iter_bisect=200):
    """使用预积分的二分法求解"""
    x0 = strain_hist[n-1]
    a = max(0.1, x0 * 0.5)
    b = min(100.0, x0 * 2.0)
    fa = ConstitutiveEqn_val_fast(a, p, n, t_step, I1[n], I2[n]) - sigma
    fb = ConstitutiveEqn_val_fast(b, p, n, t_step, I1[n], I2[n]) - sigma

    for _ in range(20):
        if fa * fb < 0:
            break
        if fa > 0:
            a *= 0.5
            if a < 1e-6: a = 1e-6; break
            fa = ConstitutiveEqn_val_fast(a, p, n, t_step, I1[n], I2[n]) - sigma
        else:
            b *= 2.0
            if b > 1e6: b = 1e6; break
            fb = ConstitutiveEqn_val_fast(b, p, n, t_step, I1[n], I2[n]) - sigma

    if fa * fb > 0:
        return x0

    for _ in range(max_iter_bisect):
        c = (a + b) / 2.0
        fc = ConstitutiveEqn_val_fast(c, p, n, t_step, I1[n], I2[n]) - sigma
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c
        if fa * fc < 0:
            b = c; fb = fc
        else:
            a = c; fa = fc
    return (a + b) / 2.0

@njit(cache=True)
def compute_creep_picard_numba(sigma, p, t_step, n_max, max_iter=100, tol=1e-6):
    strain = np.ones(n_max + 1)
    strain[0] = solve_initial_lambda(sigma, p)
    for i in range(1, n_max + 1):
        strain[i] = strain[0] + (strain[0] * 0.01) * i * t_step

    for k in range(max_iter):
        I1, I2 = precompute_integrals(strain, p, t_step, n_max)
        new_strain = np.zeros(n_max + 1)
        new_strain[0] = strain[0]
        for n in range(1, n_max + 1):
            new_strain[n] = solve_current_step_fast(strain, n, p, t_step, sigma, I1, I2)
        diff = np.max(np.abs(new_strain - strain))
        strain = new_strain

        # 每 10 次迭代或收敛时打印进度
        if k % 10 == 0 or diff < tol:
            print("Picard iter", k+1, "max residual =", diff)
        if diff < tol:
            print("Converged after", k+1, "iterations.")
            break
    return strain

# ==================== 4. 模型预测接口 ====================
def compute_creep_curve(sigma, p, beta, t_max, t_step=0.01):
    tau_max = beta * t_max
    n_max = int(tau_max / t_step)
    strain = compute_creep_picard_numba(sigma, p, t_step, n_max)
    t_bar = np.arange(len(strain)) * t_step
    return t_bar, strain

# ==================== 5. 主程序（三组数据对比） ====================
def main():
    # ---------- 1. 加载实验数据（三组） ----------
    filepath = os.path.join(save_path, '4d_creep_BH.xlsx')  # 假设文件名
    time_exp, stresses, strain_rates_exp = load_three_groups(filepath)

    # ---------- 2. 模型参数（固定） ----------
    p = 1.1
    G0 = 1.14      # 剪切模量 (Pa)
    beta = 12.8     # 松弛速率 (1/s)
    dt_step = 0.01   # 无量纲时间步长

    # ---------- 3. 对三组数据分别计算模型 ----------
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']   # 三条模型曲线的颜色
    markers = ['o', 's', '^']                     # 实验点标记
    group_names = ['$\\sigma = 0.001$', '$\\sigma = 0.002$', '$\\sigma = 0.005$']

    # 创建图形
    fig1, ax1 = plt.subplots(figsize=(14, 10))   # 应变图
    fig2, ax2 = plt.subplots(figsize=(14, 10))   # 应变率图

    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    ax2.set_xscale('linear')
    ax2.set_yscale('log')

    for i in range(3):
        print(f"\n====== 处理第 {i+1} 组数据 ======")
        # 取该组应力的均值作为实际应力（假设应力波动小）
        sigma_real = np.mean(stresses[i])
        sigma = sigma_real / G0
        t_max_physical = time_exp[-1]

        # 计算模型蠕变曲线
        t_bar, lam = compute_creep_curve(sigma, p, beta, t_max_physical, t_step=dt_step)
        strain_model = lam - 1.0
        t_model = t_bar / beta
        dlam_dtau = np.gradient(lam, dt_step)
        strain_rate_model = beta * dlam_dtau

        # 由实验应变率梯形积分得到实验应变（初始为0）
        strain_exp = np.zeros_like(time_exp)
        for j in range(1, len(time_exp)):
            dt = time_exp[j] - time_exp[j-1]
            strain_exp[j] = strain_exp[j-1] + 0.5 * (strain_rates_exp[i][j] + strain_rates_exp[i][j-1]) * dt

        # 对齐初始应变：该组模型的初始应变加到实验应变上
        epsilon0 = lam[0] - 1.0
        strain_exp_aligned = strain_exp + epsilon0
        print(f"第{i+1}组理论初始应变 ε₀ = {epsilon0:.6f}")

        # ---------- 应变图 ----------
        ax1.plot(time_exp, strain_exp_aligned, marker=markers[i], linestyle='None',
                 color=colors[i], markersize=lines_markersize,
                 label=f'{group_names[i]} (exp)', alpha=0.7)
        ax1.plot(t_model, strain_model, '-', color='black',
                 linewidth=lines_linewidth, label=f'{group_names[i]} (model)')

        # ---------- 应变率图 ----------
        mask_exp = (strain_rates_exp[i] > 0) & (time_exp > 0)
        ax2.plot(time_exp[mask_exp], strain_rates_exp[i][mask_exp], marker=markers[i],
                 linestyle='None', color=colors[i], markersize=lines_markersize,
                 label=f'{group_names[i]} (exp)', alpha=0.7)
        mask_mod = (strain_rate_model > 0) & (t_model > 0)
        ax2.plot(t_model[mask_mod], strain_rate_model[mask_mod], '-', color='black',
                 linewidth=lines_linewidth, label=f'{group_names[i]} (model)')

    # ---------- 应变图格式 ----------
    ax1.set_xlabel('Time $t$', fontsize=label_fontsize)
    ax1.set_ylabel('Strain $\\epsilon = \\lambda - 1$', fontsize=label_fontsize)
    ax1.set_title('Creep strain vs. time', fontsize=title_fontsize, pad=20)
    ax1.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)

    ax1.set_xlim(0, 10.0)

    ax1.tick_params(axis='both', which='major', direction=xtick_direction,
                    top=xtick_top, right=ytick_right, bottom=True, left=True,
                    width=xtick_major_width, length=xtick_major_size, labelsize=tick_fontsize)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='minor', direction=xtick_direction,
                    top=xtick_top, right=ytick_right, bottom=True, left=True,
                    width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)
    plt.figure(fig1.number)
    plt.tight_layout()
    fig1_name = os.path.join(save_path, "Pre_creep_strain_three_groups.png")
    plt.savefig(fig1_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变对比图已保存至 {fig1_name}")

    # ---------- 应变率图格式 ----------
    ax2.set_xlabel('Time $t$', fontsize=label_fontsize)
    ax2.set_ylabel('Strain rate $d\\lambda/dt$ (s$^{-1}$)', fontsize=label_fontsize)
    ax2.set_title('Creep strain rate vs. time', fontsize=title_fontsize, pad=20)
    ax2.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)

    ax2.set_xlim(0.009, 10.0)

    ax2.tick_params(axis='both', which='major', direction=xtick_direction,
                    top=xtick_top, right=ytick_right, bottom=True, left=True,
                    width=xtick_major_width, length=xtick_major_size, labelsize=tick_fontsize)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='minor', direction=xtick_direction,
                    top=xtick_top, right=ytick_right, bottom=True, left=True,
                    width=xtick_major_width*0.75, length=xtick_major_size*0.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)
    plt.figure(fig2.number)
    plt.tight_layout()
    fig2_name = os.path.join(save_path, "Pre_creep_strain_rate_three_groups.png")
    plt.savefig(fig2_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变率对比图已保存至 {fig2_name}")


if __name__ == "__main__":
    main()