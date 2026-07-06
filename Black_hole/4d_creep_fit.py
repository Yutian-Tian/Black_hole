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

# ==================== 1. 全局绘图风格（沿用您之前的设定） ====================
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

# 结果保存路径（请修改为您想要的目录）
save_path = "/home/tyt/project/Black_hole/4d_creep_results"
os.makedirs(save_path, exist_ok=True)

def load_data(filepath):
    """
    自动识别 CSV 或 Excel 文件，读取前两列。
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath, engine='openpyxl')  # .xlsx 用 openpyxl
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    strain_rate = df.iloc[:, 0].values 
    stress = df.iloc[:, 1].values

    return strain_rate, stress

# ==================== 2. 优化后的 Numba 核心（预计算积分，消除内循环） ====================
@njit(cache=True)
def solve_initial_lambda(sigma, p, tol=1e-10, max_iter=100):
    x = 1.0
    for _ in range(max_iter):
        f = x**(p-1) - x**(-p-1) - sigma
        if abs(f) < tol:
            break
        df = (p-1)*x**(p-2) + (p+1)*x**(-p-2)
        x = x - f/df
        if x < 1e-3:
            x = 1e-3
    return x

@njit(cache=True)
def ConstitutiveEqn_val_fast(lam_current, p, n, t_step, I1_n, I2_n):
    """
    残差计算，I1_n 和 I2_n 为预计算的积分值。
    等式：exp(-n*t_step)*(λ_n^{p-1} - λ_n^{-p-1}) + λ_n^{p-1}*I1_n - λ_n^{-p-1}*I2_n = sigma
    """
    if n == 0:
        return lam_current**(p-1) - lam_current**(-p-1)
    term1 = np.exp(-n * t_step) * (lam_current**(p-1) - lam_current**(-p-1))
    term2 = lam_current**(p-1) * I1_n - lam_current**(-p-1) * I2_n
    return term1 + term2

@njit(cache=True)
def precompute_integrals(strain_hist, p, t_step, n_max):
    """
    基于上一次迭代的应变历史 strain_hist[0..n_max] 计算积分系数：
    I1[n] = ∫_0^{n*dt} e^{-(n*dt - s)} λ(s)^{-p} ds
    I2[n] = ∫_0^{n*dt} e^{-(n*dt - s)} λ(s)^{p} ds
    采用梯形积分，充分利用历史值。
    """
    I1 = np.zeros(n_max + 1)
    I2 = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            exp_i = np.exp(-(n - i) * t_step)
            exp_i1 = np.exp(-(n - (i+1)) * t_step)
            # λ^{-p}
            lam_inv_i = strain_hist[i]**(-p)
            lam_inv_i1 = strain_hist[i+1]**(-p)
            # λ^{p}
            lam_pow_i = strain_hist[i]**p
            lam_pow_i1 = strain_hist[i+1]**p
            s1 += 0.5 * (exp_i * lam_inv_i + exp_i1 * lam_inv_i1) * t_step
            s2 += 0.5 * (exp_i * lam_pow_i + exp_i1 * lam_pow_i1) * t_step
        I1[n] = s1
        I2[n] = s2
    return I1, I2

@njit(cache=True)
def solve_current_step_fast(strain_hist, n, p, t_step, sigma, I1, I2, tol=1e-8, max_iter_bisect=100):
    """
    使用预计算的 I1[n], I2[n] 快速求解当前步 λ_n。
    """
    x0 = strain_hist[n-1]
    a = x0
    b = x0 * 2.0
    fa = ConstitutiveEqn_val_fast(a, p, n, t_step, I1[n], I2[n]) - sigma
    fb = ConstitutiveEqn_val_fast(b, p, n, t_step, I1[n], I2[n]) - sigma

    # 扩大上界直到异号
    while fa * fb > 0 and b < 1e6:
        b *= 2.0
        fb = ConstitutiveEqn_val_fast(b, p, n, t_step, I1[n], I2[n]) - sigma
    if fa * fb > 0:
        a = x0
        b = 1e6
        fa = ConstitutiveEqn_val_fast(a, p, n, t_step, I1[n], I2[n]) - sigma
        fb = ConstitutiveEqn_val_fast(b, p, n, t_step, I1[n], I2[n]) - sigma
        if fa * fb > 0:
            return b

    # 二分法
    for _ in range(max_iter_bisect):
        c = (a + b) / 2.0
        fc = ConstitutiveEqn_val_fast(c, p, n, t_step, I1[n], I2[n]) - sigma
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
def compute_creep_picard_numba_fast(sigma, p, t_step, n_max, max_iter=40, tol=1e-8):
    """
    Picard 迭代求解整个蠕变历史，每次迭代预计算 I1, I2 以加速。
    """
    strain = np.ones(n_max + 1)
    strain[0] = solve_initial_lambda(sigma, p)
    # 初始猜测
    for i in range(1, n_max + 1):
        strain[i] = strain[0] * np.exp(i * t_step * 0.01)

    for _ in range(max_iter):
        # 预计算积分系数（基于上一轮应变）
        I1, I2 = precompute_integrals(strain, p, t_step, n_max)
        new_strain = np.zeros(n_max + 1)
        new_strain[0] = strain[0]
        for n in range(1, n_max + 1):
            new_strain[n] = solve_current_step_fast(
                strain, n, p, t_step, sigma, I1, I2
            )
        diff = np.max(np.abs(new_strain - strain))
        strain = new_strain
        if diff < tol:
            break
    return strain

# ==================== 3. 模型预测接口 ====================
def compute_creep_curve(sigma, p, t_step=0.01, n_max=2000):
    try:
        strain = compute_creep_picard_numba_fast(sigma, p, t_step, n_max)
        t_bar = np.arange(len(strain)) * t_step
        return t_bar, strain
    except Exception:
        return np.full(n_max+1, np.nan), np.full(n_max+1, np.nan)

def model_strain_rate(t_exp, p, sigma_real, G0, beta):
    sigma = sigma_real / G0
    t_max_exp = np.max(t_exp)
    tau_max = beta * t_max_exp
    t_step = 0.01
    n_max = max(int(tau_max / t_step) + 500, 2000)
    t_bar, lam = compute_creep_curve(sigma, p, t_step, n_max)
    if np.any(np.isnan(lam)):
        return np.full_like(t_exp, np.inf)
    dlam_dtau = np.gradient(lam, t_step)
    strain_rate_tau = beta * dlam_dtau
    t_model = t_bar / beta
    interp_func = interp1d(t_model, strain_rate_tau, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    sr_model = interp_func(t_exp)
    mask = np.isnan(sr_model)
    if np.any(mask):
        f = interp1d(t_model, strain_rate_tau, kind='nearest',
                     bounds_error=False, fill_value='extrapolate')
        sr_model[mask] = f(t_exp[mask])
    return sr_model

# ==================== 4. 参数拟合函数 ====================
def fit_parameters(time_exp, strain_rate_exp, sigma_real,
                   p0_guess, G0_guess, beta_guess,
                   fix_p=False, bounds=None):
    if fix_p:
        def residual(x):
            G0, beta = x
            sr_model = model_strain_rate(time_exp, p0_guess, sigma_real, G0, beta)
            return (sr_model - strain_rate_exp) / np.maximum(np.abs(strain_rate_exp), 1e-6)
        x0 = np.array([G0_guess, beta_guess])
        if bounds is None:
            bounds = ([1e-12, 1e-12], [np.inf, np.inf])
    else:
        def residual(x):
            p, G0, beta = x
            sr_model = model_strain_rate(time_exp, p, sigma_real, G0, beta)
            return (sr_model - strain_rate_exp) / np.maximum(np.abs(strain_rate_exp), 1e-6)
        x0 = np.array([p0_guess, G0_guess, beta_guess])
        if bounds is None:
            bounds = ([1e-12, 1e-12, 1e-12], [10.0, np.inf, np.inf])

    print("开始参数拟合...")
    t_start = time.time()
    result = least_squares(residual, x0, bounds=bounds, method='trf',
                           ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=200, verbose=2)
    t_end = time.time()
    print(f"拟合完成，耗时 {t_end - t_start:.2f} 秒")
    print(result)

    if fix_p:
        fitted = {'p': p0_guess, 'G0': result.x[0], 'beta': result.x[1]}
    else:
        fitted = {'p': result.x[0], 'G0': result.x[1], 'beta': result.x[2]}
    return result, fitted

# ==================== 5. 主程序 ====================
def main():
    # ---------- 请根据您的实验数据修改以下部分 ----------
    filepath = os.path.join(save_path, '4d_creep_BH.xlsx')  # 请修改为实际路径
    strain_rate_exp, stress = load_data(filepath)
    sigma_real = 1e-3               # 实际应力，请按实验设置
    time_exp = np.linspace(0, 10, len(strain_rate_exp))  # 根据数据调整时间

    p_guess = 1.1
    G0_guess = 1.1
    beta_guess = 10.0

    result, params = fit_parameters(time_exp, strain_rate_exp, sigma_real,
                                    p_guess, G0_guess, beta_guess, fix_p=False)
    print("\n拟合结果：")
    for k, v in params.items():
        print(f"{k} = {v:.6e}")

    param_file = os.path.join(save_path, "fitted_parameters.txt")
    with open(param_file, 'w') as f:
        f.write("Fitted parameters:\n")
        for k, v in params.items():
            f.write(f"{k} = {v:.6e}\n")
    print(f"参数已保存至 {param_file}")

    # ---------- 绘制拟合对比图 ----------
    sr_model = model_strain_rate(time_exp, params['p'], sigma_real,
                                 params['G0'], params['beta'])

    fig, ax = plt.subplots(figsize=(14, 10))

    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#000000', '#1f77b4']
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(time_exp, strain_rate_exp, 'o', color='#d62728',
            markersize=lines_markersize, label='Experiment', linewidth=0)
    ax.plot(time_exp, sr_model, '-', color='#1f77b4',
            linewidth=lines_linewidth, label='Model fit')

    ax.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax.set_ylabel('Strain rate (s$^{-1}$)', fontsize=label_fontsize)
    ax.set_title('Creep strain rate fitting', fontsize=title_fontsize, pad=20)

    ax.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)

    ax.set_xlim([1e-2, 1e1])
    ax.set_ylim([1e-3, 1e1])

    ax.tick_params(axis='both', which='major',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width,
                   length=xtick_major_size, labelsize=tick_fontsize)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='minor',
                   direction=xtick_direction, top=xtick_top, right=ytick_right,
                   bottom=True, left=True, width=xtick_major_width * 0.75,
                   length=xtick_major_size * 0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)

    plt.tight_layout()

    fig_name = os.path.join(save_path, "4dcreep_fit_comparison.png")
    plt.savefig(fig_name, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"拟合对比图已保存至 {fig_name}")

    # ===================== 6. 新增：应变率-时间图（双对数坐标） =====================
    fig2, ax2 = plt.subplots(figsize=(14, 10))

    ax2.set_yscale('log')

    for i, (sigma, t, strain, strain_rate) in enumerate(all_curves):
        # 移除时间零点附近的可能非正值，避免对数坐标警告
        mask = (strain_rate > 0) & (t > 0)
        ax2.plot(t[mask], strain_rate[mask],
                   color=colors[i % len(colors)],
                   linewidth=lines_linewidth,
                   label=f'$\\sigma_0 = {sigma} G_0$')

    ax2.set_xlabel(f'Scaled time $\\beta t$', fontsize=label_fontsize)
    ax2.set_ylabel(f'Strain rate $d\\lambda/dt$', fontsize=label_fontsize)
    ax2.set_title(f'Creep strain rate of Vitrimers', fontsize=title_fontsize, pad=20)

    ax2.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=grid_alpha, linewidth=grid_linewidth)

    # 刻度样式（与第一张图一致）
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

    fig_name2 = os.path.join(save_path, "creep_strain_rate_numba.png")
    plt.savefig(fig_name2, dpi=savefig_dpi, bbox_inches='tight', facecolor='white')
    print(f"✅ 应变率图已保存至 {fig_name2}")

if __name__ == "__main__":
    main()