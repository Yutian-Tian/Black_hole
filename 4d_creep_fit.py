"""
creep_fitter.py — 2D聚合物网络参数拟合器
==============================================

功能：
    - 读取实验数据（多组不同应力水平）
    - 拟合全局参数 (μ, p, β)
    - 拟合策略：关注后60%数据（指数增长段）
    - 可视化拟合结果（应变率、应变、拟合误差）

拟合模型：
    全局三参数模型：
        μ: 共享弹性模量
        p: 共享本构指数  
        β: 共享链交换速率（时间缩放）

作者：基于4d_creep_robust_v2.py改写
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
from numba import njit
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局绘图设置 ====================
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_family = 'Times New Roman'
title_fontsize = 35
label_fontsize = 35
tick_fontsize = 35
legend_fontsize = 25
lines_linewidth = 4
markersize = 15

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.family'] = font_family

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'axes.titlesize': title_fontsize,
    'axes.labelsize': label_fontsize,
    'xtick.labelsize': tick_fontsize,
    'ytick.labelsize': tick_fontsize,
    'legend.fontsize': legend_fontsize,
    'axes.linewidth': 2,
    'lines.linewidth': lines_linewidth,
    'lines.markersize': markersize,
    'figure.dpi': 100,
    'savefig.dpi': 300,
})

# ==================== 保存路径 ====================
save_path = "/home/tyt/project/Black_hole/4d_creep_results"
os.makedirs(save_path, exist_ok=True)

# ====================================================================
#  核心求解器模块 (Numba-accelerated)
# ====================================================================

@njit(cache=True)
def solve_initial_lambda(sigma, p, tol=1e-12, max_iter=50):
    """求解初始瞬时弹性响应 λ₀"""
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
    """计算泰勒展开系数"""
    l0 = lambda0
    pm1, pp1 = p - 1.0, p + 1.0
    f0 = l0 ** pm1 - l0 ** (-pp1)
    fp0 = pm1 * l0 ** (p - 2) + pp1 * l0 ** (-p - 2)
    fpp0 = pm1 * (p - 2) * l0 ** (p - 3) - pp1 * (p + 2) * l0 ** (-p - 3)
    A = beta * f0 / fp0
    B = (A / 2.0) * (beta - A * fpp0 / fp0 - 2.0 * p * beta / (l0 * l0 * fp0))
    return A, B

@njit(cache=True)
def build_initial_guess(lambda0, A, B, sigma, p, beta, n_max, t_step):
    """构建初始猜测"""
    strain = np.zeros(n_max + 1)
    strain[0] = lambda0
    if sigma < 0.01:
        for i in range(1, n_max + 1):
            t = i * t_step
            strain[i] = lambda0 + A * t + B * t * t
    else:
        alpha_short = np.exp(-17.0 / (1.0 + 17.0 * sigma * sigma))
        c = alpha_short * sigma
        alpha_long = beta / (p - 1.0)
        half_al2 = 0.5 * alpha_long * alpha_long
        for i in range(1, n_max + 1):
            t = i * t_step
            strain[i] = (lambda0 - c + (A - alpha_long * c) * t 
                         + (B - half_al2 * c) * t * t + c * np.exp(alpha_long * t))
    return strain

@njit(cache=True)
def compute_creep_gauss_seidel(sigma, p=2.0, beta=1.0, t_step=0.01, n_max=4000,
                                max_iter=30, tol=1e-12, omega=0.7):
    """Gauss-Seidel Picard迭代求解蠕变本构方程"""
    lambda0 = solve_initial_lambda(sigma, p)
    A, B = compute_taylor_coeffs(lambda0, p, beta)
    strain = build_initial_guess(lambda0, A, B, sigma, p, beta, n_max, t_step)

    S1 = np.zeros(n_max + 1)
    S2 = np.zeros(n_max + 1)

    exp_dt = np.exp(-beta * t_step)
    dtau = 1.0 - exp_dt

    for k in range(max_iter):
        max_diff = 0.0
        S1[0], S2[0] = 0.0, 0.0

        for n in range(1, n_max + 1):
            tau_n = n * t_step
            exp_tn = np.exp(-beta * tau_n)

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

# ====================================================================
#  数据加载模块
# ====================================================================

def load_multi_group_data(filepath):
    """
    读取多组实验数据
    
    文件格式：
        - 每两列为一组数据：应变率, 应力
        - 时间轴假设为 0 到 10 均匀分布
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    # 假设时间轴为 0-10
    t = np.linspace(0, 10, len(df))
    groups = []
    
    for i in range(0, df.shape[1], 2):
        if i + 1 >= df.shape[1]:
            break
        eps_dot = df.iloc[:, i].values.astype(np.float64)
        stress_val = np.mean(df.iloc[:, i + 1].values)
        groups.append({
            't': t,
            'strain_rate': eps_dot,
            'stress': stress_val
        })
    
    return groups

# ====================================================================
#  模型预测与拟合模块
# ====================================================================

def model_predict_3params(t_exp, sigma_real, mu, p, beta, t_step_base=0.01, max_tau=50):
    """
    使用全局三参数模型预测蠕变曲线
    
    时间映射：
        无量纲时间 τ = β × t_physical
        求解器内部使用 β=1，通过调整 t_step 实现
    """
    sigma_dimless = sigma_real / mu
    if sigma_dimless <= 0: sigma_dimless = 1e-12
    
    # 物理时间 -> 无量纲时间
    tau_max = beta * t_exp[-1]
    if tau_max > max_tau: tau_max = max_tau
    
    # 计算求解器时间步长
    dt_phys = t_exp[1] - t_exp[0] if len(t_exp) > 1 else 0.1
    solver_t_step = beta * dt_phys
    
    # 粗化时间步以提高效率
    if solver_t_step < t_step_base:
        solver_t_step = t_step_base
        
    n_max = int(tau_max / solver_t_step)
    if n_max < 2: n_max = 2

    # 运行求解器
    lam = compute_creep_gauss_seidel(
        sigma_dimless, p=p, beta=1.0, t_step=solver_t_step, n_max=n_max,
        max_iter=30, tol=1e-12, omega=0.7
    )

    tau_arr = np.arange(len(lam)) * solver_t_step
    rate_tau = compute_strain_rate(lam, solver_t_step)
    
    # 无量纲时间 -> 物理时间
    t_model = tau_arr / beta
    # 物理应变率
    rate_phys = rate_tau * beta

    # 插值到实验时间点
    strain_model = np.interp(t_exp, t_model, lam)
    rate_model = np.interp(t_exp, t_model, rate_phys)
    
    return strain_model, rate_model


def compute_log_l2_loss_tail(rate_model, rate_exp, tail_ratio=0.6):
    """
    计算后 (tail_ratio)% 数据的 Log-L2 损失
    关注指数增长段的拟合质量
    """
    n = len(rate_exp)
    split_idx = int(n * (1.0 - tail_ratio))
    
    rate_exp_tail = rate_exp[split_idx:]
    rate_model_tail = rate_model[split_idx:]
    
    mask = (rate_exp_tail > 0) & (rate_model_tail > 0)
    if np.sum(mask) < 3:
        return 1e10
    
    log_diff = np.log(rate_model_tail[mask]) - np.log(rate_exp_tail[mask])
    return np.sum(log_diff ** 2)


def objective_3params(params, groups, t_step_base, tail_ratio):
    """
    三参数目标函数
    
    params = [mu, p, beta]
    """
    mu, p, beta = params
    
    # 参数约束
    if mu <= 1e-9 or p < 1.01 or beta <= 1e-6:
        return 1e12

    total_loss = 0.0
    for g in groups:
        try:
            _, rate_model = model_predict_3params(
                g['t'], g['stress'], mu, p, beta, t_step_base=t_step_base
            )
            loss = compute_log_l2_loss_tail(rate_model, g['strain_rate'], tail_ratio=tail_ratio)
            total_loss += loss
        except Exception:
            total_loss += 1e10
            
    return total_loss


def fit_global_parameters(groups, t_step_base=0.02, tail_ratio=0.6,
                          mu_init=0.1, p_init=2.0, beta_init=1.0):
    """
    拟合全局三参数 (μ, p, β)
    
    拟合策略：
        - 使用多个起始点避免局部最优
        - 使用 L-BFGS-B 优化算法
        - 仅拟合后 tail_ratio 的数据
    """
    print("\n" + "=" * 60)
    print(f"全局三参数拟合（关注后{int(tail_ratio*100)}%数据）")
    print("=" * 60)
    
    # 参数边界
    bounds = [
        (1e-4, 10.0),     # μ: 弹性模量
        (1.01, 10.0),     # p: 本构指数
        (1e-3, 100.0)     # β: 链交换速率
    ]
    
    x0 = [mu_init, p_init, beta_init]
    
    # 确保起始点在边界内
    for i in range(3):
        lo, hi = bounds[i]
        x0[i] = max(lo * 1.1, min(x0[i], hi * 0.9))

    best_result = None
    best_loss = np.inf
    
    # 多个起始点以提高鲁棒性
    start_points = [
        [mu_init, p_init, beta_init],
        [mu_init*0.1, p_init*0.8, beta_init*2.0],
        [mu_init*2.0, p_init*1.2, beta_init*0.5]
    ]
    
    for i, x0_try in enumerate(start_points):
        print(f"  尝试 {i+1}: 初始=[{x0_try[0]:.3f}, {x0_try[1]:.3f}, {x0_try[2]:.3f}]")
        try:
            res = minimize(
                objective_3params,
                x0_try,
                args=(groups, t_step_base, tail_ratio),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-7, 'maxiter': 200}
            )
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res
            print(f"    结果: Loss={res.fun:.4f}, μ={res.x[0]:.4f}, p={res.x[1]:.4f}, β={res.x[2]:.4f}")
        except Exception as e:
            print(f"    失败: {e}")

    if best_result:
        print("\n  最佳全局参数:")
        print(f"    μ   = {best_result.x[0]:.6f}")
        print(f"    p   = {best_result.x[1]:.6f}")
        print(f"    β   = {best_result.x[2]:.6f}")
        print(f"    Loss = {best_result.fun:.6f}")
        
        return {
            'mu': best_result.x[0],
            'p': best_result.x[1],
            'beta': best_result.x[2],
            'loss': best_result.fun,
            'success': best_result.success
        }
    else:
        print("  拟合失败！")
        return None

# ====================================================================
#  可视化模块
# ====================================================================

def plot_fitting_results(groups, fit_result, save_path, tail_ratio=0.6):
    """
    绘制拟合结果
    
    包括：
        - 应变速率对比（log-log）
        - 应变对比（log-linear）
        - 拟合误差分析
    """
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    
    color_cycle = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    # ========== 子图1: 应变速率 ==========
    ax_rate = fig.add_subplot(gs[0])
    
    for i, g in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        t_exp = g['t']
        eps_exp = g['strain_rate']
        
        # 模型预测
        strain_model, rate_model = model_predict_3params(
            t_exp, g['stress'], 
            fit_result['mu'], fit_result['p'], fit_result['beta'],
            t_step_base=0.01
        )
        
        # 实验数据
        mask_exp = eps_exp > 0
        ax_rate.plot(t_exp[mask_exp], eps_exp[mask_exp], 'o', 
                     color=color, markersize=8, alpha=0.6,
                     label=f'Exp $\\sigma$={g["stress"]:.1e}')
        
        # 模型预测
        mask_mod = rate_model > 0
        ax_rate.plot(t_exp[mask_mod], rate_model[mask_mod], '-', 
                     color=color, linewidth=3,
                     label='Model')
        
        # 标记拟合区域
        split_idx = int(len(t_exp) * (1.0 - tail_ratio))
        if split_idx < len(t_exp):
            ax_rate.axvline(t_exp[split_idx], color=color, linestyle=':', alpha=0.5)
            ax_rate.axvspan(t_exp[split_idx], t_exp[-1], color=color, alpha=0.05)
    
    ax_rate.set_yscale('log')
    ax_rate.set_ylabel('Strain Rate $d\\lambda/dt$', fontsize=label_fontsize)
    ax_rate.set_title(f'Strain Rate Fit ($\\mu={fit_result["mu"]:.3f}, p={fit_result["p"]:.3f}, \\beta={fit_result["beta"]:.3f}$)\n'
                      f'Optimized on last {int(tail_ratio*100)}% (shaded)', fontsize=title_fontsize, pad=20)
    ax_rate.legend(fontsize=legend_fontsize*0.8, loc='best', ncol=2)
    ax_rate.grid(True, alpha=0.3)
    
    # ========== 子图2: 应变 ==========
    ax_strain = fig.add_subplot(gs[1])
    
    for i, g in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        t_exp = g['t']
        eps_exp = g['strain_rate']
        
        # 计算实验应变（梯形积分）
        strain_exp = np.zeros_like(t_exp)
        for k in range(1, len(t_exp)):
            dt = t_exp[k] - t_exp[k-1]
            strain_exp[k] = strain_exp[k-1] + 0.5 * (eps_exp[k] + eps_exp[k-1]) * dt
        
        # 模型预测
        strain_model, rate_model = model_predict_3params(
            t_exp, g['stress'], 
            fit_result['mu'], fit_result['p'], fit_result['beta'],
            t_step_base=0.01
        )
        strain_mod = strain_model - 1.0
        
        # 绘图
        mask_se = strain_exp > 0
        ax_strain.plot(t_exp[mask_se], strain_exp[mask_se], 'o',
                       color=color, markersize=8, alpha=0.6,
                       label=f'Exp $\\sigma$={g["stress"]:.1e}')
        mask_sm = strain_mod > 0
        ax_strain.plot(t_exp[mask_sm], strain_mod[mask_sm], '-',
                       color=color, linewidth=3,
                       label='Model')
        
        # 标记拟合区域
        split_idx = int(len(t_exp) * (1.0 - tail_ratio))
        if split_idx < len(t_exp):
            ax_strain.axvline(t_exp[split_idx], color=color, linestyle=':', alpha=0.5)
    
    ax_strain.set_ylabel('Strain $\\lambda-1$', fontsize=label_fontsize)
    ax_strain.set_title('Strain Fit', fontsize=title_fontsize, pad=20)
    ax_strain.legend(fontsize=legend_fontsize*0.8, loc='best', ncol=2)
    ax_strain.grid(True, alpha=0.3)
    
    # ========== 子图3: 拟合误差 ==========
    ax_err = fig.add_subplot(gs[2])
    
    for i, g in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        t_exp = g['t']
        eps_exp = g['strain_rate']
        
        # 模型预测
        strain_model, rate_model = model_predict_3params(
            t_exp, g['stress'], 
            fit_result['mu'], fit_result['p'], fit_result['beta'],
            t_step_base=0.01
        )
        
        # 计算相对误差（对数空间）
        mask = (eps_exp > 0) & (rate_model > 0)
        log_err = np.log(rate_model[mask]) - np.log(eps_exp[mask])
        rel_err = np.abs(log_err) / np.abs(np.log(eps_exp[mask])) * 100
        
        ax_err.plot(t_exp[mask], rel_err, 'o-', 
                    color=color, markersize=6, linewidth=2, alpha=0.7,
                    label=f'$\\sigma$={g["stress"]:.1e}')
        
        # 标记拟合区域
        split_idx = int(len(t_exp) * (1.0 - tail_ratio))
        if split_idx < len(t_exp):
            ax_err.axvline(t_exp[split_idx], color=color, linestyle=':', alpha=0.5)
    
    ax_err.set_xlabel('Time $t$', fontsize=label_fontsize)
    ax_err.set_ylabel('Relative Error (%) (log scale)', fontsize=label_fontsize)
    ax_err.set_title('Fitting Error Analysis', fontsize=title_fontsize, pad=20)
    ax_err.legend(fontsize=legend_fontsize*0.8, loc='best')
    ax_err.grid(True, alpha=0.3)
    ax_err.set_ylim(0, ax_err.get_ylim()[1] * 1.1)
    
    # 保存图形
    filepath = os.path.join(save_path, "fitting_results.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  拟合结果图已保存: {filepath}")


def plot_parameter_comparison(fit_result, save_path, true_params=None):
    """
    绘制参数对比图
    """
    if true_params is None:
        return
    
    params = ['μ', 'p', 'β']
    labels = ['$\\mu$', '$p$', '$\\beta$']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(params))
    width = 0.35
    
    true_vals = [true_params['mu'], true_params['p'], true_params['beta']]
    fit_vals = [fit_result['mu'], fit_result['p'], fit_result['beta']]
    
    bars1 = ax.bar(x - width/2, true_vals, width, label='True', 
                   color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x + width/2, fit_vals, width, label='Fitted', 
                   color='#d62728', alpha=0.8)
    
    ax.set_ylabel('Parameter Value', fontsize=label_fontsize)
    ax.set_title('Parameter Comparison: True vs Fitted', fontsize=title_fontsize, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=20)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, "parameter_comparison.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  参数对比图已保存: {filepath}")

# ====================================================================
#  生成合成数据（用于测试）
# ====================================================================

def generate_synthetic_data(n_groups=3, noise_level=0.1):
    """
    生成合成实验数据用于测试拟合程序
    
    参数：
        n_groups: 数据组数
        noise_level: 噪声水平
    """
    print("\n生成合成实验数据...")
    
    # 真实参数
    true_params = {'mu': 0.5, 'p': 2.0, 'beta': 2.0}
    
    # 不同的应力水平
    sigma_values = [0.05, 0.1, 0.2][:n_groups]
    
    # 时间轴
    t = np.logspace(-2, 1, 100)  # 0.01 to 10
    df_data = {}
    
    for i, sigma in enumerate(sigma_values):
        # 计算理论蠕变曲线
        sigma_dimless = sigma / true_params['mu']
        strain = compute_creep_gauss_seidel(
            sigma_dimless, p=true_params['p'], beta=true_params['beta'],
            t_step=0.01, n_max=2000
        )
        tau = np.arange(len(strain)) * 0.01
        rate = compute_strain_rate(strain, 0.01)
        
        # 插值到实验时间点
        rate_interp = np.interp(true_params['beta'] * t, tau, rate)
        
        # 添加噪声
        noise = np.random.normal(0, noise_level * rate_interp, len(rate_interp))
        rate_noisy = rate_interp * (1 + noise)
        
        # 确保正值
        rate_noisy = np.maximum(rate_noisy, rate_interp * 0.5)
        
        df_data[f'rate_{i}'] = rate_noisy
        df_data[f'stress_{i}'] = np.full(len(t), sigma)
    
    df = pd.DataFrame(df_data)
    
    # 保存
    filepath = os.path.join(save_path, 'synthetic_experimental_data.xlsx')
    df.to_excel(filepath, index=False)
    print(f"  合成数据已保存: {filepath}")
    print(f"  真实参数: μ={true_params['mu']}, p={true_params['p']}, β={true_params['beta']}")
    
    return filepath, true_params

# ====================================================================
#  主程序
# ====================================================================

def main():
    """主程序：读取数据、拟合参数、可视化结果"""
    print("=" * 60)
    print("2D聚合物网络参数拟合器")
    print("=" * 60)
    
    # ----------------------- 1. 数据加载 -----------------------
    data_path = os.path.join(save_path, '4d_creep_BH.xlsx')
    
    # 如果实验数据不存在，生成合成数据
    if not os.path.exists(data_path):
        print(f"\n实验数据文件不存在: {data_path}")
        print("生成合成数据用于测试...")
        data_path, true_params = generate_synthetic_data(n_groups=3, noise_level=0.1)
    else:
        true_params = None
        print(f"\n加载实验数据: {data_path}")
    
    groups = load_multi_group_data(data_path)
    print(f"成功加载 {len(groups)} 组数据")
    
    for i, g in enumerate(groups):
        print(f"  组 {i+1}: σ = {g['stress']:.4e}, "
              f"应变率范围 = [{g['strain_rate'].min():.2e}, {g['strain_rate'].max():.2e}]")
    
    # ----------------------- 2. 参数拟合 -----------------------
    fit_res = fit_global_parameters(
        groups, 
        t_step_base=0.02, 
        tail_ratio=0.6,  # 拟合后60%数据
        mu_init=0.5, 
        p_init=2.0, 
        beta_init=1.0
    )
    
    if not fit_res:
        print("拟合失败，退出程序")
        return
    
    # ----------------------- 3. 保存拟合参数 -----------------------
    params_path = os.path.join(save_path, "fitted_params.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(f"# 拟合参数\n")
        f.write(f"# Global 3-Parameter Fit (μ, p, β)\n")
        f.write(f"# Fitting strategy: Last 60% of data\n\n")
        f.write(f"mu   = {fit_res['mu']:.8e}    # Elastic modulus\n")
        f.write(f"p    = {fit_res['p']:.8e}    # Constitutive exponent\n")
        f.write(f"beta = {fit_res['beta']:.8e}    # Chain exchange rate\n")
        f.write(f"loss = {fit_res['loss']:.8e}\n")
        f.write(f"success = {fit_res['success']}\n")
    
    if true_params:
        with open(params_path, 'a', encoding='utf-8') as f:
            f.write(f"\n# 真实参数（合成数据）\n")
            f.write(f"mu_true   = {true_params['mu']:.8e}\n")
            f.write(f"p_true    = {true_params['p']:.8e}\n")
            f.write(f"beta_true = {true_params['beta']:.8e}\n")
    
    print(f"\n拟合参数已保存: {params_path}")
    
    # ----------------------- 4. 可视化拟合结果 -----------------------
    print("\n生成拟合结果可视化...")
    plot_fitting_results(groups, fit_res, save_path, tail_ratio=0.6)
    
    # 如果是合成数据，绘制参数对比图
    if true_params:
        plot_parameter_comparison(fit_res, save_path, true_params)
    
    # ----------------------- 5. 生成预测数据集 -----------------------
    print("\n生成模型预测数据...")
    df_pred = pd.DataFrame()
    
    for i, g in enumerate(groups):
        strain_model, rate_model = model_predict_3params(
            g['t'], g['stress'], 
            fit_res['mu'], fit_res['p'], fit_res['beta'],
            t_step_base=0.01
        )
        
        df_pred[f't_group{i+1}'] = g['t']
        df_pred[f'rate_exp_group{i+1}'] = g['strain_rate']
        df_pred[f'rate_model_group{i+1}'] = rate_model
        df_pred[f'strain_exp_group{i+1}'] = np.cumsum(g['strain_rate']) * (g['t'][1]-g['t'][0])
        df_pred[f'strain_model_group{i+1}'] = strain_model - 1.0
    
    pred_path = os.path.join(save_path, "model_predictions.csv")
    df_pred.to_csv(pred_path, index=False, float_format='%.8e')
    print(f"预测数据已保存: {pred_path}")
    
    print("\n" + "=" * 60)
    print("拟合完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
