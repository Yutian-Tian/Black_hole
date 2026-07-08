"""
预积分加速的蠕变模型（拟合应变率数据）
将 p, G0, beta 作为拟合参数，拟合实验应变率-时间曲线
"""

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
    """Picard 迭代求解全部时间步，返回 λ 数组"""
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
        if diff < tol:
            break
    return strain

# ==================== 4. 模型预测接口 ====================
def compute_creep_curve(sigma, p, beta, t_max, t_step=0.01):
    """返回无量纲时间 t_bar 和拉伸 λ 数组"""
    tau_max = beta * t_max
    n_max = int(tau_max / t_step)
    strain = compute_creep_picard_numba(sigma, p, t_step, n_max)
    t_bar = np.arange(len(strain)) * t_step
    return t_bar, strain

# ==================== 5. 拟合目标函数 ====================
def residuals(params, time_exp, strain_rate_exp, sigma_real, t_step):
    """
    计算模型应变率与实验应变率的残差
    params: [p, G0, beta]
    """
    p, G0, beta = params
    # 无量纲应力
    sigma = sigma_real / G0
    # 计算模型蠕变曲线（时间范围略大于实验）
    t_max = time_exp[-1] * 1.05
    try:
        t_bar, lam = compute_creep_curve(sigma, p, beta, t_max, t_step)
    except Exception as e:
        # 若数值求解失败，返回大残差
        return np.ones_like(strain_rate_exp) * 1e6
    strain_model = lam - 1.0
    t_model = t_bar / beta
    # 应变率：dλ/dt = β * dλ/dτ
    dlam_dtau = np.gradient(lam, t_step)
    strain_rate_model = beta * dlam_dtau
    # 插值到实验时间点
    interp_func = interp1d(t_model, strain_rate_model, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    model_rate_interp = interp_func(time_exp)
    return model_rate_interp - strain_rate_exp

# ==================== 6. 主程序（拟合 + 绘图） ====================
def main():
    # ---------- 1. 加载实验数据 ----------
    filepath = os.path.join(save_path, '4d_creep_BH.xlsx')
    strain_rate_exp, stress = load_data(filepath)
    time_exp = np.linspace(0, 10, len(strain_rate_exp))

    if not np.all(np.diff(time_exp) > 0):
        raise ValueError("时间序列应单调递增，请检查数据顺序")

    # 梯形积分得到实验应变（供后续对比使用）
    strain_exp = np.zeros_like(time_exp)
    for i in range(1, len(time_exp)):
        dt = time_exp[i] - time_exp[i-1]
        strain_exp[i] = strain_exp[i-1] + 0.5 * (strain_rate_exp[i] + strain_rate_exp[i-1]) * dt

    # ---------- 2. 设定实验应力、计算步长、初始参数 ----------
    # 假设实验施加的应力为常数，这里取应力列的平均值或沿用原值 1e-3 Pa
    sigma_real = np.mean(stress) if np.std(stress) < 1e-9 else 1e-3  # 根据数据情况选择
    dt_step = 0.1   # 无量纲时间步长（平衡精度与速度）

    # 初始猜测 [p, G0, beta]
    x0 = np.array([1.1, 1.14, 3.2])
    # 参数边界（全部为正）
    bounds = ([1.0, 0.0, 0.0], [10.0, 100.0, 100.0])

    print("开始拟合参数 p, G0, beta ...")
    t_start = time.time()

    # ---------- 3. 非线性最小二乘拟合 ----------
    result = least_squares(
        residuals, x0,
        args=(time_exp, strain_rate_exp, sigma_real, dt_step),
        bounds=bounds,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=200,
        verbose=2
    )

    t_end = time.time()
    print(f"拟合完成，耗时 {t_end - t_start:.1f} 秒")
    print(f"最优参数: p = {result.x[0]:.6f}, G0 = {result.x[1]:.6f} Pa, beta = {result.x[2]:.6f} 1/s")
    print(f"残差范数: {result.cost:.6e}")

    # ---------- 4. 用最优参数计算模型完整曲线 ----------
    p_opt, G0_opt, beta_opt = result.x
    sigma_opt = sigma_real / G0_opt
    t_max_physical = time_exp[-1]
    t_bar_opt, lam_opt = compute_creep_curve(sigma_opt, p_opt, beta_opt, 
                                             t_max_physical, t_step=dt_step)
    strain_model = lam_opt - 1.0
    t_model = t_bar_opt / beta_opt
    dlam_dtau = np.gradient(lam_opt, dt_step)
    strain_rate_model = beta_opt * dlam_dtau

    # 对齐初始应变：将理论 t=0 时的应变加到实验数据上
    epsilon0 = lam_opt[0] - 1.0
    strain_exp_aligned = strain_exp + epsilon0
    print(f"理论初始应变 ε₀ = {epsilon0:.6f}，已将实验应变整体上移该值。")

    # ---------- 5. 应变-时间图（线性x，对数y） ----------
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    ax1.set_xscale('linear')
    ax1.set_yscale('log')
    
    ax1.plot(time_exp, strain_exp_aligned, 'o', color='#d62728', 
             markersize=lines_markersize, label='Experiment (aligned)', linewidth=0)
    ax1.plot(t_model, strain_model, '-', color='#1f77b4', 
             linewidth=lines_linewidth, label='Model (fitted)')
    
    ax1.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax1.set_ylabel('Strain $\\epsilon = \\lambda - 1$', fontsize=label_fontsize)
    ax1.set_title('Creep strain vs time ($\\sigma = 0.001$)', fontsize=title_fontsize, pad=20)
    ax1.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)

    ax1.set_xlim(0, 10.0)
    
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
    fig1_name = os.path.join(save_path, "Pre_creep_strain_fitted.png")
    plt.savefig(fig1_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变对比图已保存至 {fig1_name}")

    # ---------- 6. 应变率-时间图（双对数） ----------
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    mask_exp = (strain_rate_exp > 0) & (time_exp > 0)
    ax2.plot(time_exp[mask_exp], strain_rate_exp[mask_exp], 'o', 
             color='#d62728', markersize=lines_markersize, label='Experiment (rate)', linewidth=0)
    mask_mod = (strain_rate_model > 0) & (t_model > 0)
    ax2.plot(t_model[mask_mod], strain_rate_model[mask_mod], '-', 
             color='#1f77b4', linewidth=lines_linewidth, label='Model (fitted rate)')
    
    ax2.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax2.set_ylabel('Strain rate $d\\lambda/dt$ (s$^{-1}$)', fontsize=label_fontsize)
    ax2.set_title('Creep strain rate vs time ($\\sigma = 0.001$)', fontsize=title_fontsize, pad=20)
    ax2.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    
    ax1.set_xlim(0.009, 10.0)

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
    fig2_name = os.path.join(save_path, "Pre_creep_strain_rate_fitted.png")
    plt.savefig(fig2_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变率对比图已保存至 {fig2_name}")

if __name__ == "__main__":
    main()