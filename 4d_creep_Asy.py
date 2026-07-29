import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from numba import njit
from scipy.optimize import least_squares
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

save_path = "/home/tyt/project/Black_hole/4d_creep_results"
os.makedirs(save_path, exist_ok=True)

# ==================== 2. 核心工具函数 ====================
@njit(cache=True)
def solve_initial_lambda(sigma, p, tol=1e-12, max_iter=100):
    x = 1.0 + sigma / (2 * p)
    for _ in range(max_iter):
        f = x**(p-1) - x**(-p-1) - sigma
        if abs(f) < tol:
            break
        df = (p-1)*x**(p-2) + (p+1)*x**(-p-2)
        x = x - f / df
        if x < 1e-6:
            x = 1e-6
    return x

# ==================== 3. 短时间渐近解析解 ====================
def creep_asymptotic(t, sigma_real, mu, p, beta):
    sigma_nd = sigma_real / mu
    lambda0 = solve_initial_lambda(sigma_nd, p)
    f_prime_nd = (p - 1) * lambda0**(p - 2) + (p + 1) * lambda0**(-(p + 2))
    f_double_prime_nd = (p - 1)*(p - 2)*lambda0**(p - 3) - (p + 1)*(p + 2)*lambda0**(-(p + 3))
    A = beta * sigma_nd / f_prime_nd
    term1 = beta
    term2 = A * f_double_prime_nd / f_prime_nd
    term3 = 2 * p * beta / (lambda0**2 * f_prime_nd)
    B = (A / 2) * (term1 - term2 - term3)
    lambda_t = lambda0 + A * t + B * t**2
    strain_t = lambda_t - 1.0
    strain_rate_t = A + 2 * B * t
    return lambda_t, strain_t, strain_rate_t

# ==================== 4. 实验数据加载 ====================
def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    strain_rate_exp = df.iloc[:, 0].values
    time_exp = np.linspace(0, 10.0, len(strain_rate_exp))
    strain_exp = np.zeros_like(time_exp)
    for i in range(1, len(time_exp)):
        dt = time_exp[i] - time_exp[i-1]
        strain_exp[i] = strain_exp[i-1] + 0.5 * (strain_rate_exp[i] + strain_rate_exp[i-1]) * dt
    return time_exp, strain_exp, strain_rate_exp

# ==================== 5. 拟合函数 ====================
def model_strain_rate(t, sigma_real, p, mu, beta):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    _, _, strain_rate = creep_asymptotic(t, sigma_real, mu, p, beta)
    return strain_rate

def fit_asymptotic_to_data(time_exp, rate_exp, sigma_real,
                           p0=1.3, mu0=5e-3, beta0=1.0,
                           fit_fraction=0.1, use_log_residuals=True,
                           bounds=None):
    if bounds is None:
        bounds = ([1.001, 1e-12, 1e-12], [10.0, 1e12, 1e12])
    t_max_fit = time_exp.max() * fit_fraction
    fit_mask = time_exp <= t_max_fit
    t_fit = time_exp[fit_mask]
    r_fit = rate_exp[fit_mask]
    if len(t_fit) < 5:
        raise ValueError("可用于拟合的数据点太少，请减小 fit_fraction 或检查数据")
    def residuals(params, t, r_exp, sigma, use_log):
        p, mu, beta = params
        model_rate = model_strain_rate(t, sigma, p, mu, beta)
        if use_log:
            safe_model = np.where(model_rate > 0, model_rate, 1e-20)
            safe_exp = np.where(r_exp > 0, r_exp, 1e-20)
            return np.log10(safe_model) - np.log10(safe_exp)
        else:
            return model_rate - r_exp
    initial_params = np.array([p0, mu0, beta0])
    try:
        result = least_squares(
            residuals,
            initial_params,
            bounds=bounds,
            args=(t_fit, r_fit, sigma_real, use_log_residuals),
            max_nfev=1000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12
        )
        popt = result.x
        popt[0] = np.clip(popt[0], bounds[0][0], bounds[1][0])
        popt[1] = np.clip(popt[1], bounds[0][1], bounds[1][1])
        popt[2] = np.clip(popt[2], bounds[0][2], bounds[1][2])
    except Exception as e:
        print(f"拟合失败：{e}，将使用初始猜测值")
        popt = initial_params
    return popt, fit_mask

# ==================== 6. 主程序（已修改：对齐初始应变） ====================
def main():
    # ---------- 1. 加载实验数据 ----------
    filepath = os.path.join(save_path, '4d_creep_BH.xlsx')
    time_exp, strain_exp, strain_rate_exp = load_data(filepath)
    
    if not np.all(np.diff(time_exp) > 0):
        raise ValueError("时间序列应单调递增，请检查数据顺序")
    if np.any(strain_rate_exp <= 0):
        print("警告：实验应变率中存在非正值，拟合时将忽略这些点")
    
    # ---------- 2. 实验条件与拟合设置 ----------
    sigma_real = 1e-3           # 实验恒应力 (Pa)
    p_init = 1.3
    G0_init = 5e-3              # 剪切模量 μ (Pa)
    beta_init = 1.0             # 松弛速率 (1/s)
    fit_fraction = 0.4
    use_log_fit = True
    custom_bounds = ([0.0, 1e-6, 1e-6], [5.0, 1e3, 1e3])
    
    # ---------- 3. 执行拟合 ----------
    print("正在进行参数拟合（使用最小二乘法 + 参数边界约束）...")
    popt, fit_mask = fit_asymptotic_to_data(
        time_exp, strain_rate_exp, sigma_real,
        p0=p_init, mu0=G0_init, beta0=beta_init,
        fit_fraction=fit_fraction,
        use_log_residuals=use_log_fit,
        bounds=custom_bounds
    )
    p_opt, G0_opt, beta_opt = popt
    print(f"\n拟合完成！")
    print(f"  拟合区间: 0 ~ {time_exp[fit_mask].max():.3f} s")
    print(f"  拟合参数: p = {p_opt:.4f}, G0 = {G0_opt:.4e} Pa, beta = {beta_opt:.4e} s⁻¹")
    
    # ---------- 4. 对齐初始应变：将实验应变上移理论初始应变 ----------
    lambda0_opt = solve_initial_lambda(sigma_real / G0_opt, p_opt)
    epsilon0_theory = lambda0_opt - 1.0
    strain_exp_aligned = strain_exp + epsilon0_theory
    print(f"理论初始应变 ε₀ = {epsilon0_theory:.6f}，已将实验应变整体上移该值。")
    
    # ---------- 5. 用拟合参数计算渐近解 ----------
    t_asy = np.linspace(0, time_exp.max(), 1000)
    _, strain_asy, strain_rate_asy = creep_asymptotic(
        t_asy, sigma_real, G0_opt, p_opt, beta_opt
    )
    
    # ---------- 6. 应变-时间图（对数y） ----------
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    ax1.set_yscale('log')
    
    ax1.plot(time_exp, strain_exp_aligned, 'o', color='#d62728',
             markersize=lines_markersize, label='Experiment (aligned)', linewidth=0)
    ax1.plot(t_asy, strain_asy, '--', color='#2ca02c',
             linewidth=lines_linewidth, label='Asymptotic fit ($O(t^2)$)')
    
    t_fit_max = time_exp[fit_mask].max()
    ax1.axvspan(0, t_fit_max, color='gray', alpha=0.1, label='Fitting range')
    
    ax1.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax1.set_ylabel('Strain $\\varepsilon = \\lambda - 1$', fontsize=label_fontsize)
    ax1.set_title('Creep strain vs time (initial strain aligned)', fontsize=title_fontsize, pad=20)
    ax1.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)
    
    y_min = min(strain_exp_aligned[strain_exp_aligned>0].min(), strain_asy[strain_asy>0].min()) * 0.8
    y_max = max(strain_exp_aligned.max(), strain_asy.max()) * 1.5
    ax1.set_ylim([y_min, y_max])
    ax1.set_xlim([0, max(time_exp.max(), t_asy.max()) * 1.05])
    
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
    fig1_path = os.path.join(save_path, "Asy_fitted_strain_aligned.png")
    plt.savefig(fig1_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变对比图已保存至: {fig1_path}")
    plt.close()

    # ---------- 7. 应变率-时间图（双对数） ----------
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    mask_pos_exp = (time_exp > 0) & (strain_rate_exp > 0)
    ax2.plot(time_exp[mask_pos_exp], strain_rate_exp[mask_pos_exp], 'o',
             color='#d62728', markersize=lines_markersize, label='Experiment', linewidth=0)
    mask_pos_asy = (t_asy > 0) & (strain_rate_asy > 0)
    ax2.plot(t_asy[mask_pos_asy], strain_rate_asy[mask_pos_asy], '--',
             color='#2ca02c', linewidth=lines_linewidth, label='Asymptotic fit ($O(t^2)$)')
    
    ax2.axvspan(1e-3, t_fit_max, color='gray', alpha=0.1, label='Fitting range')
    
    ax2.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax2.set_ylabel('Strain rate $d\\varepsilon/dt$ (s$^{-1}$)', fontsize=label_fontsize)
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
    fig2_path = os.path.join(save_path, "Asy_fitted_strain_rate_comparison.png")
    plt.savefig(fig2_path, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变率对比图已保存至: {fig2_path}")

if __name__ == "__main__":
    main()