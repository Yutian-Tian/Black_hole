import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
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

# ==================== 2. 数据加载函数 ====================
def load_data(filepath):
    """读取第一列为应变率，第二列为应力（本示例仅用第一列）"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    strain_rate = df.iloc[:, 0].values 
    stress = df.iloc[:, 1].values   # 备用

    return strain_rate, stress

# ==================== 3. 核心物理函数 ====================
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
def ConstitutiveEqn_val(strain_hist, t_idx, p, t_step, current_val):
    if current_val < 1e-6:
        current_val = 1e-6
    if t_idx == 0:
        return current_val**(p-1) - current_val**(-p-1)
    
    term1 = np.exp(-t_idx * t_step) * (current_val**(p-1) - current_val**(-p-1))
    term2 = 0.0
    for i in range(t_idx):
        exp_i = np.exp(-(t_idx - i) * t_step)
        lam_i = max(strain_hist[i], 1e-6)
        A_i = exp_i * (current_val**(p-1) / lam_i**p - lam_i**p / current_val**(p+1))
        
        exp_i1 = np.exp(-(t_idx - (i+1)) * t_step)
        lam_i1 = max(strain_hist[i+1], 1e-6)
        A_i1 = exp_i1 * (current_val**(p-1) / lam_i1**p - lam_i1**p / current_val**(p+1))
        
        term2 += 0.5 * (A_i + A_i1) * t_step
    return term1 + term2

@njit(cache=True)
def solve_current_step(strain_hist, n, p, t_step, sigma, tol=1e-8, max_iter_bisect=200):
    x0 = strain_hist[n-1]
    a = max(0.1, x0 * 0.5)
    b = min(100.0, x0 * 2.0)
    fa = ConstitutiveEqn_val(strain_hist, n, p, t_step, a) - sigma
    fb = ConstitutiveEqn_val(strain_hist, n, p, t_step, b) - sigma
    
    for _ in range(20):
        if fa * fb < 0:
            break
        if fa > 0:
            a *= 0.5
            if a < 1e-6:
                a = 1e-6
                break
            fa = ConstitutiveEqn_val(strain_hist, n, p, t_step, a) - sigma
        else:
            b *= 2.0
            if b > 1e6:
                b = 1e6
                break
            fb = ConstitutiveEqn_val(strain_hist, n, p, t_step, b) - sigma
    
    if fa * fb > 0:
        return x0   # 未找到根，保持原值
    
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
def compute_creep_picard_numba(sigma, p, t_step, n_max, max_iter=100, tol=1e-6):
    strain = np.ones(n_max + 1)
    strain[0] = solve_initial_lambda(sigma, p)
    for i in range(1, n_max + 1):
        strain[i] = strain[0] + (strain[0] * 0.01) * i * t_step
    
    for k in range(max_iter):
        new_strain = np.zeros(n_max + 1)
        new_strain[0] = strain[0]
        for n in range(1, n_max + 1):
            new_strain[n] = solve_current_step(strain, n, p, t_step, sigma)
        
        diff = np.max(np.abs(new_strain - strain))
        strain = new_strain
        
        # 每 10 次迭代或收敛时打印一次
        if k % 10 == 0 or diff < tol:
            print("Picard iter", k+1, "max residual =", diff)
        
        if diff < tol:
            print("Converged after", k+1, "iterations.")
            break
    
    return strain

# ==================== 4. 模型预测接口（改为按时间范围求解） ====================
def compute_creep_curve(sigma, p, beta, t_max, t_step=0.01):
    """
    根据物理时间范围计算蠕变曲线。
    :param sigma: 无量纲应力
    :param p: 材料参数
    :param beta: 松弛速率 (1/s)
    :param t_max: 求解的物理时间上限 (s)
    :param t_step: 无量纲时间步长（默认0.01）
    :return: t_bar (无量纲时间), strain (拉伸比λ)
    """
    tau_max = beta * t_max  # 无量纲时间上限
    n_max = int(tau_max / t_step)  # 所需步数
    strain = compute_creep_picard_numba(sigma, p, t_step, n_max)
    t_bar = np.arange(len(strain)) * t_step
    return t_bar, strain

# ==================== 5. 主程序 ====================
def main():
    # ---------- 1. 加载实验数据 ----------
    filepath = os.path.join(save_path, '4d_creep_BH.xlsx')
    strain_rate_exp, stress = load_data(filepath)
    time_exp = np.linspace(0, 10, len(strain_rate_exp))

    if not np.all(np.diff(time_exp) > 0):
        raise ValueError("时间序列应单调递增，请检查数据顺序")

    # 梯形积分得到实验应变
    strain_exp = np.zeros_like(time_exp)
    for i in range(1, len(time_exp)):
        dt = time_exp[i] - time_exp[i-1]
        strain_exp[i] = strain_exp[i-1] + 0.5 * (strain_rate_exp[i] + strain_rate_exp[i-1]) * dt

    # ---------- 2. 模型参数 ----------
    params = {
        'p': 1.2,
        'G0': 6.13,   # 剪切模量 μ (Pa)
        'beta': 59.5   # 松弛速率 (1/s)
    }
    sigma_real = 1e-3       # 实际应力 (Pa)
    dt_step = 0.01          # 无量纲时间步长（可调节精度）

    # ---------- 3. 计算模型曲线（覆盖实验全时段） ----------
    sigma = sigma_real / params['G0']
    t_max_physical = time_exp[-1]  # 确保求解到实验结束时刻
    t_bar, lam = compute_creep_curve(sigma, params['p'], params['beta'], 
                                     t_max_physical, t_step=dt_step)
    strain_model = lam - 1.0
    t_model = t_bar / params['beta']
    dlam_dtau = np.gradient(lam, dt_step)
    strain_rate_model = params['beta'] * dlam_dtau

    # ---------- 4. 应变-时间图（线性x，对数y） ----------
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    
    ax1.plot(time_exp, strain_exp, 'o', color='#d62728', 
             markersize=lines_markersize, label='Experiment (strain)', linewidth=0)
    ax1.plot(t_model, strain_model, '-', color='#1f77b4', 
             linewidth=lines_linewidth, label='Model (strain)')
    
    ax1.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax1.set_ylabel('Strain $\\epsilon = \\lambda - 1$', fontsize=label_fontsize)
    ax1.set_title('Creep strain vs time', fontsize=title_fontsize, pad=20)
    ax1.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    
    ax1.tick_params(axis='both', which='major',
                    direction=xtick_direction, top=xtick_top, right=ytick_right,
                    bottom=True, left=True, width=xtick_major_width,
                    length=xtick_major_size, labelsize=tick_fontsize)
    ax1.minorticks_on()
    ax1.tick_params(axis='both', which='minor',
                    direction=xtick_direction, top=xtick_top, right=ytick_right,
                    bottom=True, left=True, width=xtick_major_width * 0.75,
                    length=xtick_major_size * 0.5)
    for spine in ax1.spines.values():
        spine.set_linewidth(axes_linewidth)
    plt.tight_layout()
    fig1_name = os.path.join(save_path, "creep_strain_comparison.png")
    plt.savefig(fig1_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变对比图已保存至 {fig1_name}")

    # ---------- 5. 应变率-时间图（双对数） ----------
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    mask_exp = (strain_rate_exp > 0) & (time_exp > 0)
    ax2.plot(time_exp[mask_exp], strain_rate_exp[mask_exp], 'o', 
             color='#d62728', markersize=lines_markersize, label='Experiment (rate)', linewidth=0)
    mask_mod = (strain_rate_model > 0) & (t_model > 0)
    ax2.plot(t_model[mask_mod], strain_rate_model[mask_mod], '-', 
             color='#1f77b4', linewidth=lines_linewidth, label='Model (rate)')
    
    ax2.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax2.set_ylabel('Strain rate $d\\lambda/dt$ (s$^{-1}$)', fontsize=label_fontsize)
    ax2.set_title('Creep strain rate vs time', fontsize=title_fontsize, pad=20)
    ax2.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    
    ax2.tick_params(axis='both', which='major',
                    direction=xtick_direction, top=xtick_top, right=ytick_right,
                    bottom=True, left=True, width=xtick_major_width,
                    length=xtick_major_size, labelsize=tick_fontsize)
    ax2.minorticks_on()
    ax2.tick_params(axis='both', which='minor',
                    direction=xtick_direction, top=xtick_top, right=ytick_right,
                    bottom=True, left=True, width=xtick_major_width * 0.75,
                    length=xtick_major_size * 0.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(axes_linewidth)
    plt.tight_layout()
    fig2_name = os.path.join(save_path, "creep_strain_rate_comparison.png")
    plt.savefig(fig2_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变率对比图已保存至 {fig2_name}")

if __name__ == "__main__":
    main()