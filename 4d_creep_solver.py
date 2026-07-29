"""
creep_solver.py — 2D聚合物网络蠕变求解器
==========================================================

功能：
    - 给定无量纲应力 (σ̄)、材料参数 (p, β)，求解蠕变行为
    - 可视化应变 λ(τ) 和应变率 λ̇(τ)
    - 输出数值数据到CSV文件
    - 参数敏感性分析

作者：基于4d_creep_robust.py改写
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
from numba import njit
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
    """
    使用牛顿法求解初始瞬时弹性响应 λ₀
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
    计算泰勒展开系数 A, B
    λ(τ) ≈ λ₀ + A·τ + B·τ²
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
def build_initial_guess(lambda0, A, B, sigma, p, beta, n_max, t_step):
    """
    构建初始猜测
    小应力：纯多项式
    大应力：多项式 + 指数项
    """
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
    """
    Gauss-Seidel Picard迭代求解蠕变本构方程
    
    方程：
        σ̄ = e^{-τ}·f(λ(τ)) + ∫₀^τ dτ' e^{-(τ-τ')}·g(λ(τ), λ(τ'))
    
    参数：
        sigma  : 无量纲应力 σ̄
        p      : 本构指数
        beta   : 链交换速率（时间缩放因子）
        t_step : 无量纲时间步长
        n_max  : 最大时间步数
    """
    lambda0 = solve_initial_lambda(sigma, p)
    A, B = compute_taylor_coeffs(lambda0, p, beta)
    strain = build_initial_guess(lambda0, A, B, sigma, p, beta, n_max, t_step)

    S1 = np.zeros(n_max + 1)
    S2 = np.zeros(n_max + 1)

    exp_dt = np.exp(-beta * t_step)
    dtau = 1.0 - exp_dt  # 精确积分因子

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
    """
    前向差分计算应变率 λ̇ = dλ/dτ
    """
    n = len(strain)
    rate = np.zeros(n)
    rate[0] = (strain[1] - strain[0]) / t_step
    for i in range(1, n):
        rate[i] = (strain[i] - strain[i - 1]) / t_step
    return rate

# ====================================================================
#  主求解函数
# ====================================================================

def solve_creep_for_stresses(sigma_list, p=2.0, beta=1.0, t_step=0.01, n_max=5000):
    """
    批量求解多个应力下的蠕变行为
    
    返回：
        results: 列表，每个元素为 (sigma, tau, strain, strain_rate)
    """
    results = []
    print(f"\n求解蠕变行为: p={p}, β={beta}, n_max={n_max}")
    print("-" * 50)
    
    for sigma in sigma_list:
        start_t = time.time()
        strain = compute_creep_gauss_seidel(
            sigma, p=p, beta=beta, t_step=t_step, n_max=n_max
        )
        tau = np.arange(len(strain)) * t_step
        rate = compute_strain_rate(strain, t_step)
        end_t = time.time()
        
        results.append({
            'sigma': sigma,
            'tau': tau,
            'strain': strain,
            'strain_rate': rate
        })
        
        print(f"  σ̄={sigma:8.4f}: 最终λ={strain[-1]:.4f}, 最终λ̇={rate[-1]:.4e}, 耗时={end_t-start_t:.2f}s")
    
    return results

# ====================================================================
#  可视化函数
# ====================================================================

def plot_strain_rate(results, save_path, filename="strain_rate.png"):
    """
    绘制应变率曲线（log-log）
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        ax.plot(res['tau'], res['strain_rate'], 
                color=color, linewidth=3,
                label=f'$\\bar{{\\sigma}} = {res["sigma"]}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimensionless Time $\\tau$', fontsize=label_fontsize)
    ax.set_ylabel('Strain Rate $d\\lambda/d\\tau$', fontsize=label_fontsize)
    ax.set_title('Creep Strain Rate Evolution', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize, loc='best')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  应变率图已保存: {filepath}")

def plot_strain(results, save_path, filename="strain.png"):
    """
    绘制应变曲线（log-linear）
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        ax.plot(res['tau'], res['strain'] - 1.0, 
                color=color, linewidth=3,
                label=f'$\\bar{{\\sigma}} = {res["sigma"]}$')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Dimensionless Time $\\tau$', fontsize=label_fontsize)
    ax.set_ylabel('Creep Strain $\\lambda - 1$', fontsize=label_fontsize)
    ax.set_title('Creep Strain Evolution', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize, loc='best')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  应变图已保存: {filepath}")

def plot_combined(results, save_path, filename="combined_creep.png"):
    """
    组合绘制应变和应变率（上下两个子图）
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 18))
    colors = ['#7b2d8e', '#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    
    # 应变率
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        axes[0].plot(res['tau'], res['strain_rate'], 
                     color=color, linewidth=3,
                     label=f'$\\bar{{\\sigma}} = {res["sigma"]}$')
    
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Strain Rate $d\\lambda/d\\tau$', fontsize=label_fontsize)
    axes[0].legend(fontsize=legend_fontsize, loc='best')
    axes[0].grid(True, linestyle=':', alpha=0.3)
    
    # 应变
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        axes[1].plot(res['tau'], res['strain'] - 1.0, 
                     color=color, linewidth=3)
    
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Dimensionless Time $\\tau$', fontsize=label_fontsize)
    axes[1].set_ylabel('Creep Strain $\\lambda - 1$', fontsize=label_fontsize)
    axes[1].grid(True, linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  组合图已保存: {filepath}")

def save_results_to_csv(results, save_path, filename="creep_results.csv"):
    """
    保存结果到CSV文件
    """
    df_list = []
    for res in results:
        df = pd.DataFrame({
            f'sigma_{res["sigma"]}_tau': res['tau'],
            f'sigma_{res["sigma"]}_strain': res['strain'],
            f'sigma_{res["sigma"]}_strain_rate': res['strain_rate']
        })
        df_list.append(df)
    
    df_all = pd.concat(df_list, axis=1)
    filepath = os.path.join(save_path, filename)
    df_all.to_csv(filepath, index=False, float_format='%.8e')
    print(f"  数据已保存: {filepath}")

# ====================================================================
#  参数敏感性分析
# ====================================================================

def parameter_sensitivity_analysis():
    """
    参数敏感性分析：不同 μ, p, β 下的应变率
    """
    print("\n" + "=" * 60)
    print("参数敏感性分析")
    print("=" * 60)
    
    # 应力范围
    sigma_list = np.array([0.01, 0.05, 0.1])
    
    # μ值范围（影响σ̄ = σ/μ，这里用不同的sigma范围模拟μ的影响）
    mu_values = np.logspace(-2, 0, 3)  # 0.01, 0.1, 1.0
    
    # p值范围
    p_values = [1.5, 2.0, 3.0]
    
    # β值范围
    beta_values = [0.5, 1.0, 2.0]
    
    # 1. μ敏感性（通过sigma范围模拟）
    fig_mu, axes_mu = plt.subplots(1, len(mu_values), figsize=(20, 6))
    for i, mu in enumerate(mu_values):
        ax = axes_mu[i]
        sigma_scaled = sigma_list / mu
        
        for sigma in sigma_scaled:
            strain = compute_creep_gauss_seidel(sigma, p=2.0, beta=1.0, t_step=0.01, n_max=3000)
            tau = np.arange(len(strain)) * 0.01
            rate = compute_strain_rate(strain, 0.01)
            ax.plot(tau, rate, linewidth=2, label=f'$\\bar{{\\sigma}}={sigma:.3f}$')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$\\tau$', fontsize=label_fontsize*0.7)
        ax.set_ylabel('$d\\lambda/d\\tau$', fontsize=label_fontsize*0.7)
        ax.set_title(f'$\\mu = {mu:.2f}$', fontsize=title_fontsize*0.8)
        ax.legend(fontsize=15)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_mu.savefig(os.path.join(save_path, "sensitivity_mu.png"), dpi=300)
    plt.close()
    print("  μ敏感性图已保存")
    
    # 2. p敏感性
    fig_p, ax_p = plt.subplots(figsize=(14, 10))
    sigma = 0.1
    for p in p_values:
        strain = compute_creep_gauss_seidel(sigma, p=p, beta=1.0, t_step=0.01, n_max=3000)
        tau = np.arange(len(strain)) * 0.01
        rate = compute_strain_rate(strain, 0.01)
        ax_p.plot(tau, rate, linewidth=3, label=f'$p = {p}$')
    
    ax_p.set_xscale('log')
    ax_p.set_yscale('log')
    ax_p.set_xlabel('$\\tau$', fontsize=label_fontsize)
    ax_p.set_ylabel('$d\\lambda/d\\tau$', fontsize=label_fontsize)
    ax_p.set_title('p-Sensitivity (Fixed $\\bar{{\\sigma}}=0.1$)', fontsize=title_fontsize)
    ax_p.legend(fontsize=legend_fontsize)
    ax_p.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_p.savefig(os.path.join(save_path, "sensitivity_p.png"), dpi=300)
    plt.close()
    print("  p敏感性图已保存")
    
    # 3. β敏感性
    fig_beta, ax_beta = plt.subplots(figsize=(14, 10))
    sigma = 0.1
    for beta in beta_values:
        strain = compute_creep_gauss_seidel(sigma, p=2.0, beta=beta, t_step=0.01, n_max=3000)
        tau = np.arange(len(strain)) * 0.01
        rate = compute_strain_rate(strain, 0.01)
        ax_beta.plot(tau, rate, linewidth=3, label=f'$\\beta = {beta}$')
    
    ax_beta.set_xscale('log')
    ax_beta.set_yscale('log')
    ax_beta.set_xlabel('$\\tau$', fontsize=label_fontsize)
    ax_beta.set_ylabel('$d\\lambda/d\\tau$', fontsize=label_fontsize)
    ax_beta.set_title('$\\beta$-Sensitivity (Fixed $\\bar{{\\sigma}}=0.1, p=2$)', fontsize=title_fontsize)
    ax_beta.legend(fontsize=legend_fontsize)
    ax_beta.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_beta.savefig(os.path.join(save_path, "sensitivity_beta.png"), dpi=300)
    plt.close()
    print("  β敏感性图已保存")

# ====================================================================
#  主程序
# ====================================================================

def main():
    """主程序：求解并可视化蠕变行为"""
    print("=" * 60)
    print("2D聚合物网络蠕变求解器")
    print("=" * 60)
    
    # ----------------------- 配置参数 -----------------------
    # 应力列表（无量纲）
    sigma_list = [0.002, 0.01, 0.02, 0.05, 0.1, 0.2]
    
    # 材料参数
    p = 2.0      # 本构指数
    beta = 1.0   # 链交换速率
    
    # 求解参数
    t_step = 0.01   # 无量纲时间步长
    n_max = 5000    # 最大时间步数
    
    # ----------------------- 求解 -----------------------
    results = solve_creep_for_stresses(
        sigma_list, p=p, beta=beta, 
        t_step=t_step, n_max=n_max
    )
    
    # ----------------------- 保存数据 -----------------------
    save_results_to_csv(results, save_path, "creep_solver_results.csv")
    
    # ----------------------- 可视化 -----------------------
    print("\n生成可视化图形...")
    plot_strain_rate(results, save_path, "solver_strain_rate.png")
    plot_strain(results, save_path, "solver_strain.png")
    plot_combined(results, save_path, "solver_combined.png")
    
    # ----------------------- 参数敏感性分析 -----------------------
    print("\n是否运行参数敏感性分析？(y/n)")
    # 取消注释以下两行以启用交互式询问
    # choice = input().strip().lower()
    # if choice == 'y':
    parameter_sensitivity_analysis()
    
    print("\n" + "=" * 60)
    print("求解完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
