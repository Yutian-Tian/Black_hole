"""
本程序基于插值公式拟合实验数据，使用显式解析公式计算应变 λ(t) 和应变率 dλ/dt。
拟合参数为 μ, p, β，λ₀ 由隐式方程 f(λ₀) = σ₀ 求解，不再作为独立拟合参数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import time
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
#  核心解析求解器模块 (基于给定的解析公式与隐式 λ₀ 定义)
# ====================================================================

def solve_elastic_lambda(sigma, mu, p, tol=1e-12, max_iter=50):
    """
    求解隐式方程 f(λ₀) = σ
    f(λ) = μ * (λ^(p-1) - λ^(-(p+1)))
    """
    if sigma <= 0.0:
        return 1.0
    
    # 初始猜测 (基于小变形下的线性近似)
    x = 1.0 + sigma / (2.0 * p * mu)
    if x < 1.0: 
        x = 1.0 + 1e-12
        
    for _ in range(max_iter):
        x_p1 = x ** (p - 1)
        x_m1 = x ** (-p - 1)
        f_val = mu * (x_p1 - x_m1) - sigma
        
        if abs(f_val) < tol:
            break
            
        # df/dλ = μ * [(p-1)λ^(p-2) + (p+1)λ^(-p-2)]
        df_val = mu * ((p - 1.0) * x ** (p - 2) + (p + 1.0) * x ** (-p - 2))
        
        x_new = x - f_val / df_val
        if x_new < 1.0:
            x_new = 1.0 + 1e-12
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
        
    return x

def compute_exact_coeffs(sigma, mu, p, beta):
    """
    根据应力 σ 计算该应力条件下的 λ₀, A 和 B 系数
    """
    # 1. 解析求解该应力下的 λ₀ (核心修改：不再作为拟合参数)
    lambda0 = solve_elastic_lambda(sigma, mu, p)
        
    # 2. 计算分母公共项 g = (p-1)λ^(p-2) + (p+1)λ^(-p-2)
    g = (p - 1.0) * lambda0 ** (p - 2.0) + (p + 1.0) * lambda0 ** (-p - 2.0)
    
    if abs(g) < 1e-15:
        return lambda0, 0.0, 0.0

    # 3. 系数 A (基于隐式约束 f(λ₀)=σ, σ/μ = λ₀^(p-1) - λ₀^(-p-1))
    numerator_A = lambda0 ** (p - 1.0) - lambda0 ** (-p - 1.0)
    A = beta * numerator_A / g

    # 4. 中间变量 f' 和 f''
    # f'(λ) = μ * g
    # f''(λ) = μ * [(p-1)(p-2)λ^(p-3) - (p+1)(p+2)λ^(-(p+3))]
    f_prime = mu * g
    f_double_prime = mu * ((p - 1.0) * (p - 2.0) * lambda0 ** (p - 3.0) - 
                           (p + 1.0) * (p + 2.0) * lambda0 ** (-p - 3.0))

    # 5. 系数 B
    term2 = A * f_double_prime / f_prime
    term3 = (2.0 * p * beta * mu) / (lambda0**2 * f_prime)
    B = (A / 2.0) * (beta - term2 - term3)
    
    return lambda0, A, B

def model_predict_3params(t_exp, sigma_real, mu, p, beta):
    """
    使用显式解析公式计算 λ(t) 和 dλ/dt (基于应力依赖的 λ₀)
    """
    if mu <= 0 or p <= 1.01 or beta <= 0:
        return np.zeros_like(t_exp), np.zeros_like(t_exp)
    
    # 根据当前应力求解该组的 λ₀, A, B
    lambda0, A, B = compute_exact_coeffs(sigma_real, mu, p, beta)

    # 指数增长项参数
    alpha = beta / (p - 1.0)
    sigma_dimless = sigma_real / mu

    # 1. 解析计算 λ(t)
    strain = lambda0 + A * t_exp + B * t_exp**2 + sigma_dimless * (np.exp(alpha * t_exp) - 1.0)

    # 2. 解析计算 dλ/dt (应变率)
    rate = A + 2.0 * B * t_exp + sigma_dimless * alpha * np.exp(alpha * t_exp)

    return strain, rate

# ====================================================================
#  数据加载模块 (保持不变)
# ====================================================================
def load_multi_group_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

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
#  模型预测与拟合模块 (恢复到三参数拟合)
# ====================================================================

def compute_log_l2_loss(rate_model, rate_exp, tail_ratio=0.6):
    n = len(rate_exp)
    split_idx = int(n * (1.0 - tail_ratio))
    
    rate_exp_tail = rate_exp[split_idx:]
    rate_model_tail = rate_model[split_idx:]
    
    mask = (rate_exp_tail > 0) & (rate_model_tail > 0)
    if np.sum(mask) < 3:
        return 1e10
    
    log_diff = np.log(rate_model_tail[mask]) - np.log(rate_exp_tail[mask])
    return np.sum(log_diff ** 2)

def objective_3params(params, groups, tail_ratio):
    """
    三参数目标函数: params = [mu, p, beta]
    (lambda0 已经由隐式方程 f(lambda0)=sigma 内部求解)
    """
    mu, p, beta = params
    
    if mu <= 1e-9 or p <= 1.01 or beta <= 1e-6:
        return 1e12

    total_loss = 0.0
    for g in groups:
        try:
            # 这里不再传入 lambda0
            _, rate_model = model_predict_3params(
                g['t'], g['stress'], mu, p, beta
            )
            loss = compute_log_l2_loss(rate_model, g['strain_rate'], tail_ratio=tail_ratio)
            total_loss += loss
        except Exception:
            total_loss += 1e10
            
    return total_loss

def fit_global_3params(groups, tail_ratio=0.6,
                       mu_init=1.0, p_init=2.0, beta_init=5.0,
                       bounds=None, start_points=None):
    print("\n" + "=" * 70)
    print(f"全局三参数拟合 (μ, p, β)，λ₀由 f(λ₀)=σ 隐式推导")
    print("=" * 70)
    
    if bounds is None:
        bounds = [
            (1e-8, 10.0),       # μ: 弹性模量
            (1.01, 5.0),        # p: 本构指数
            (1e-8, 1.0)        # β: 链交换速率
        ]
    
    x0 = [mu_init, p_init, beta_init]
    
    for i in range(3):
        lo, hi = bounds[i]
        x0[i] = max(lo * 1.1, min(x0[i], hi * 0.9))

    best_result = None
    best_loss = np.inf
    
    if start_points is None:
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
                args=(groups, tail_ratio),
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

def plot_strain_rate_fitting(groups, fit_result, save_path, tail_ratio=0.6):
    fig, ax = plt.subplots(figsize=(12, 8))
    color_cycle = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for i, g in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        t_exp = g['t']
        eps_exp = g['strain_rate']
        
        # 调用修改后的三参数预测
        strain_model, rate_model = model_predict_3params(
            t_exp, g['stress'], 
            fit_result['mu'], fit_result['p'], fit_result['beta']
        )
        
        mask_exp = eps_exp > 0
        ax.plot(t_exp[mask_exp], eps_exp[mask_exp], 'o', 
                color=color, markersize=8, alpha=0.4,
                label=f'Exp $\\sigma$={g["stress"]:.1e}')
        
        mask_mod = rate_model > 0
        ax.plot(t_exp[mask_mod], rate_model[mask_mod], '-', 
                color='black', linewidth=3,
                label='Model')
        
        split_idx = int(len(t_exp) * (1.0 - tail_ratio))
        if split_idx < len(t_exp):
            ax.axvline(t_exp[split_idx], color=color, linestyle=':', alpha=0.5)
            ax.axvspan(t_exp[split_idx], t_exp[-1], color=color, alpha=0.05)
    
    ax.set_yscale('log')
    ax.set_xlabel('Time $t$', fontsize=label_fontsize)
    ax.set_ylabel('Strain Rate $d\\lambda/dt$', fontsize=label_fontsize)
    ax.set_xlim(0.0, 10.0)
    ax.set_title(f'Strain Rate Fit\n'
                 f'Optimized on last {int(tail_ratio*100)}% (shaded)', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize*0.8, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(save_path, "strain_rate_fitting.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  应变率拟合图已保存: {filepath}")

def plot_strain_fitting(groups, fit_result, save_path, tail_ratio=0.6):
    fig, ax = plt.subplots(figsize=(12, 8))
    color_cycle = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for i, g in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        t_exp = g['t']
        eps_exp = g['strain_rate']
        
        # 积分实验数据获得应变曲线用于对比
        strain_exp = np.zeros_like(t_exp)
        for k in range(1, len(t_exp)):
            dt = t_exp[k] - t_exp[k-1]
            strain_exp[k] = strain_exp[k-1] + 0.5 * (eps_exp[k] + eps_exp[k-1]) * dt
        
        strain_model, rate_model = model_predict_3params(
            t_exp, g['stress'], 
            fit_result['mu'], fit_result['p'], fit_result['beta']
        )
        strain_mod = strain_model - 1.0
        
        mask_se = strain_exp > 0
        ax.plot(t_exp[mask_se], strain_exp[mask_se], 'o',
                color=color, markersize=8, alpha=0.6,
                label=f'Exp $\\sigma$={g["stress"]:.1e}')
        mask_sm = strain_mod > 0
        ax.plot(t_exp[mask_sm], strain_mod[mask_sm], '-',
                color='black', linewidth=3,
                label='Model')
        
        split_idx = int(len(t_exp) * (1.0 - tail_ratio))
        if split_idx < len(t_exp):
            ax.axvline(t_exp[split_idx], color=color, linestyle=':', alpha=0.5)

    ax.set_xlabel('Time $t$', fontsize=label_fontsize)
    ax.set_ylabel('Strain $\\lambda-1$', fontsize=label_fontsize)
    ax.set_xlim(0.0, 10.0)
    ax.set_title('Strain Fit', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize*0.8, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(save_path, "strain_fitting.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  应变拟合图已保存: {filepath}")

# ====================================================================
#  主程序
# ====================================================================

def main():
    """主程序：读取数据、解析模型三参数拟合 (μ, p, β)、结果可视化"""
    print("=" * 60)
    print("基于显式解析公式与隐式 λ₀ 的蠕变模型拟合器")
    print("=" * 60)

    # ================= 【参数配置区 - 全都在 main() 最上方】 =================
    fit_ratio = 0.5          # 拟合比例 (关注后 60% 的指数增长段)
    
    # 初始参数猜测点 (mu, p, beta)
    mu0 = 1e-6               # 弹性模量 μ
    p0 = 1.1                 # 本构指数 p
    beta0 = 1e-1             # 时间缩放因子 β

    # 优化搜索边界 (param_bounds) - 严格限制在合理范围内
    param_bounds = [
        (1e-8, 2.0),       # μ: 弹性模量, 物理上大于0
        (1.01, 1.5),        # p: 本构指数, 必须严格大于1
        (1e-8, 5.0)        # β: 物理时间映射系数, 大于0
    ]

    # 多个起始点以提高全局搜索鲁棒性 (start_points)
    start_points = [
        [mu0, p0, beta0],
        [mu0*0.1, p0*0.8, beta0*2.0],
        [mu0*2.0, p0*1.2, beta0*0.5]
    ]
    # ========================================================================

    # ----------------------- 1. 数据加载 -----------------------
    data_path = os.path.join(save_path, '4d_creep_BH.xlsx')
    
    if not os.path.exists(data_path):
        print(f"\n实验数据文件不存在: {data_path}")
        print("请将实验数据放置于指定路径。")
        return
    
    groups = load_multi_group_data(data_path)
    print(f"\n成功加载 {len(groups)} 组数据")
    
    for i, g in enumerate(groups):
        print(f"  组 {i+1}: σ = {g['stress']:.4e}, "
              f"应变率范围 = [{g['strain_rate'].min():.2e}, {g['strain_rate'].max():.2e}]")
    
    # ----------------------- 2. 参数拟合 -----------------------
    fit_res = fit_global_3params(
        groups, 
        tail_ratio=fit_ratio,
        mu_init=mu0, 
        p_init=p0, 
        beta_init=beta0,
        bounds=param_bounds,       # 传入参数边界
        start_points=start_points  # 传入多起点策略
    )
    
    if not fit_res:
        print("拟合失败，退出程序")
        return
    
    # ----------------------- 3. 保存拟合参数 -----------------------
    params_path = os.path.join(save_path, "fitted_params.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(f"# 显式解析蠕变模型拟合参数 (μ, p, β)\n")
        f.write(f"# λ₀ 由 f(λ₀) = σ₀ 隐式推导，非独立拟合参数\n")
        f.write(f"# Fitting strategy: Last {int(fit_ratio*100)}% of data\n\n")
        f.write(f"mu      = {fit_res['mu']:.8e}    # Elastic modulus\n")
        f.write(f"p       = {fit_res['p']:.8e}    # Constitutive exponent\n")
        f.write(f"beta    = {fit_res['beta']:.8e}    # Chain exchange rate\n")
        f.write(f"loss    = {fit_res['loss']:.8e}\n")
        f.write(f"success = {fit_res['success']}\n")

        # ========== 新增：将各组数据对应的 lambda0 写入文件 ==========
        f.write(f"\n# 各组数据对应的 λ₀ (由 f(λ₀) = σ 隐式推导)\n")
        for i, g in enumerate(groups):
            # 使用拟合后的全局 μ 和 p，计算该组应力 σ 对应的 λ₀
            lam0 = solve_elastic_lambda(g['stress'], fit_res['mu'], fit_res['p'])
            f.write(f"sigma_{i+1} = {g['stress']:.8e} \t lambda0_{i+1} = {lam0:.8f}\n")
        # ============================================================
    
    print(f"\n拟合参数已保存: {params_path}")
    
    # ========== 新增：将各组 lambda0 打印到屏幕 ==========
    print("\n各组数据对应的 λ₀ (由 f(λ₀) = σ 隐式推导):")
    for i, g in enumerate(groups):
        lam0 = solve_elastic_lambda(g['stress'], fit_res['mu'], fit_res['p'])
        print(f"  Group {i+1} (σ = {g['stress']:.4e}): λ₀ = {lam0:.8f}")
    # =====================================================
    
    # ----------------------- 4. 可视化拟合结果 -----------------------
    print("\n生成拟合结果可视化...")
    plot_strain_rate_fitting(groups, fit_res, save_path, fit_ratio)
    plot_strain_fitting(groups, fit_res, save_path, fit_ratio)
    
    # ----------------------- 5. 生成预测数据集 -----------------------
    print("\n生成模型预测数据...")
    df_pred = pd.DataFrame()
    
    for i, g in enumerate(groups):
        strain_model, rate_model = model_predict_3params(
            g['t'], g['stress'], 
            fit_res['mu'], fit_res['p'], fit_res['beta']
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