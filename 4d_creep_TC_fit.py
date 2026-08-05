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
#  核心求解器模块 
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
def compute_creep_gauss_seidel(sigma, p=2.0, a=0.5, t_step=0.01, n_max=4000,
                               max_iter=6, tol=1e-12, omega=0.7, picard_max=3,
                               verbose=False):
    """
    Picard + Newton 迭代求解图片中的本构方程
    """
    lambda0 = solve_initial_lambda(sigma, p)
    strain = np.zeros(n_max + 1)
    strain[0] = lambda0

    print_step = max(1, n_max // 10)

    for n in range(1, n_max + 1):
        if verbose and (n % print_step == 0 or n == n_max):
            print("   [Numba Solver Progress]: ", n, " / ", n_max)

        lam = strain[n - 1]

        for picard in range(picard_max):
            R = 1.0 - a + a * lam
            if R < 1e-12:
                R = 1e-12
            
            exp_R_dt = np.exp(-R * t_step)
            weight = 1.0
            
            sum_I1 = 0.0
            sum_I2 = 0.0
            
            for k in range(n - 1, -1, -1):
                integ_w = weight * (1.0 - exp_R_dt)
                lam_k = strain[k]
                sum_I1 += integ_w * (lam_k ** (-p))
                sum_I2 += integ_w * (lam_k ** p)
                weight *= exp_R_dt
            
            A_coeff = weight 
            
            lam_new = lam
            pm1, pp1 = p - 1.0, p + 1.0
            for newton in range(max_iter):
                lam_pm1 = lam_new ** pm1
                lam_mpp1 = lam_new ** (-pp1)
                
                f_val = A_coeff * (lam_pm1 - lam_mpp1) + lam_pm1 * sum_I1 - lam_mpp1 * sum_I2 - sigma
                
                if abs(f_val) < tol:
                    break
                
                df_val = (A_coeff + sum_I1) * pm1 * lam_new ** (p - 2.0) + \
                         (A_coeff + sum_I2) * pp1 * lam_new ** (-p - 2.0)
                
                dlam = f_val / df_val
                max_step = 0.5 * lam
                if dlam > max_step:
                    dlam = max_step
                elif dlam < -max_step:
                    dlam = -max_step
                
                lam_new = lam_new - dlam
                if lam_new < 1.0:
                    lam_new = 1.0 + 1e-12
            
            lam = omega * lam_new + (1.0 - omega) * lam
            if abs(lam - lam_new) < tol:
                break
                
        strain[n] = lam
    return strain

def compute_strain_rate_dimless(strain, t_step):
    """前向差分计算无量纲空间下的应变率 dλ/dτ"""
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
#  模型预测与拟合模块 (4 Parameters: μ, p, β, a)
# ====================================================================

def model_predict_4params(t_exp, sigma_real, mu, p, beta, a, solver_t_step=0.01, verbose=False):
    """
    4参数蠕变模型预测：μ, p, β, a
    【新增】动态步长机制，防止 beta 超出物理边界导致计算量爆炸。
    """
    sigma_dimless = sigma_real / mu
    if sigma_dimless <= 0:
        sigma_dimless = 1e-12

    tau_max = beta * t_exp[-1]
    
    # ========== 【核心：动态步长】强制网格点数 N <= 2000 ==========
    # 不管 beta 多大，自适应调整 dtau，保证总计算量恒定
    desired_n = 2000
    dtau = max(0.005, tau_max / desired_n)  
    # ============================================================
    
    n_max = int(np.ceil(tau_max / dtau)) + 1

    lam = compute_creep_gauss_seidel(
        sigma_dimless,
        p=p,
        a=a,
        t_step=dtau,
        n_max=n_max,
        max_iter=6,
        tol=1e-12,
        omega=0.7,
        picard_max=3,
        verbose=verbose 
    )

    tau_arr = np.arange(len(lam)) * dtau
    rate_tau = compute_strain_rate_dimless(lam, dtau)

    t_model_phys = tau_arr / beta
    rate_phys = beta * rate_tau

    strain_model = np.interp(t_exp, t_model_phys, lam)
    rate_model = np.interp(t_exp, t_model_phys, rate_phys)

    return strain_model, rate_model


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


# ========== 闭包变量，用于记录并打印优化器的迭代次数 ==========
iteration_counter = [0]

def objective_4params(params, groups, solver_t_step, tail_ratio):
    mu, p, beta, a = params
    
    if mu <= 1e-9 or p < 1.01 or beta <= 1e-6 or a < 0.0 or a > 1.0:
        return 1e12

    iteration_counter[0] += 1
    print(f"\n[拟合迭代 {iteration_counter[0]}] 尝试参数: mu={mu:.4e}, p={p:.4f}, beta={beta:.4f}, a={a:.4f}", end="")

    total_loss = 0.0
    try:
        for g in groups:
            _, rate_model = model_predict_4params(
                g['t'], g['stress'], mu, p, beta, a, solver_t_step=solver_t_step
            )
            loss = compute_log_l2_loss(rate_model, g['strain_rate'], tail_ratio=tail_ratio)
            total_loss += loss
        print(f" -> Loss = {total_loss:.4e}")
    except Exception as e:
        total_loss = 1e10
        print(f" -> 遇到异常，Loss 回退为 1e10 ({e})")
            
    return total_loss


def fit_global_parameters_4params(groups, solver_t_step=0.01, tail_ratio=0.6,
                                  mu_init=1.0, p_init=2.0, beta_init=0.05, a_init=0.5,
                                  bounds=None, start_points=None):
    """拟合全局四参数，支持外部传入 bounds 和 start_points"""
    print("\n" + "=" * 60)
    print(f"全局四参数拟合（T-Correction 无量纲模型，关注后{int(tail_ratio*100)}%数据）")
    print("=" * 60)
    
    # 如果外部没有传入 bounds，使用默认
    if bounds is None:
        bounds = [
            (1e-8, 100.0),     # μ
            (1.01, 10.0),      # p
            (1e-8, 10.0),      # β
            (0.0, 1.0)         # a
        ]
    
    x0 = [mu_init, p_init, beta_init, a_init]
    for i in range(4):
        lo, hi = bounds[i]
        x0[i] = max(lo * 1.1, min(x0[i], hi * 0.9))

    best_result = None
    best_loss = np.inf
    
    # 如果外部没有传入 start_points，使用默认
    if start_points is None:
        start_points = [
            [mu_init, p_init, beta_init, a_init],
            [mu_init*0.1, p_init*0.8, beta_init*2.0, a_init*0.2],
            [mu_init*2.0, p_init*1.2, beta_init*0.5, a_init*0.8]
        ]
    
    for i, x0_try in enumerate(start_points):
        iteration_counter[0] = 0 
        print(f"\n---> 尝试 {i+1}: 初始=[{x0_try[0]:.3f}, {x0_try[1]:.3f}, {x0_try[2]:.3f}, {x0_try[3]:.3f}]")
        try:
            res = minimize(
                objective_4params,
                x0_try,
                args=(groups, solver_t_step, tail_ratio),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-7, 'maxiter': 200}
            )
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_result = res
            print(f"    结果: Loss={res.fun:.4f}, μ={res.x[0]:.4f}, p={res.x[1]:.4f}, β={res.x[2]:.4f}, a={res.x[3]:.4f}")
        except Exception as e:
            print(f"    失败: {e}")

    if best_result:
        print("\n  最佳全局参数 (本征方程无量纲 T-Correction):")
        print(f"    μ   = {best_result.x[0]:.6f}")
        print(f"    p   = {best_result.x[1]:.6f}")
        print(f"    β   = {best_result.x[2]:.6f} (物理时间映射系数)")
        print(f"    a   = {best_result.x[3]:.6f}")
        print(f"    Loss = {best_result.fun:.6f}")
        
        return {
            'mu': best_result.x[0],
            'p': best_result.x[1],
            'beta': best_result.x[2],
            'a': best_result.x[3],
            'loss': best_result.fun,
            'success': best_result.success
        }
    else:
        print("  拟合失败！")
        return None

# ====================================================================
#  可视化模块 (保持不变)
# ====================================================================
def plot_strain_rate_fitting(groups, fit_result, save_path, tail_ratio=0.6):
    fig, ax = plt.subplots(figsize=(12, 8))
    color_cycle = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for i, g in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        t_exp = g['t']
        eps_exp = g['strain_rate']
        
        strain_model, rate_model = model_predict_4params(
            t_exp, g['stress'], 
            fit_result['mu'], fit_result['p'], fit_result['beta'], fit_result['a'],
            solver_t_step=0.01
        )
        
        mask_exp = eps_exp > 0
        ax.plot(t_exp[mask_exp], eps_exp[mask_exp], 'o', 
                color=color, markersize=8, alpha=0.4,
                label=f'Exp $\\sigma$={g["stress"]:.1e}')
        
        mask_mod = rate_model > 0
        ax.plot(t_exp[mask_mod], rate_model[mask_mod], '-', 
                color=color, linewidth=3,
                label='Model')
        
        split_idx = int(len(t_exp) * (1.0 - tail_ratio))
        if split_idx < len(t_exp):
            ax.axvline(t_exp[split_idx], color=color, linestyle=':', alpha=0.5)
            ax.axvspan(t_exp[split_idx], t_exp[-1], color=color, alpha=0.05)
    
    ax.set_yscale('log')
    ax.set_xlabel('Physical Time $t$', fontsize=label_fontsize)
    ax.set_ylabel('Strain Rate $d\\lambda/dt$', fontsize=label_fontsize)
    ax.set_xlim(0.0, 10.0)
    ax.set_title(f'Strain Rate Fit (T-Correction, $a={fit_result["a"]:.4f}$)\n'
                 f'Optimized on last {int(tail_ratio*100)}% (shaded)', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize*0.8, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(save_path, "strain_rate_fitting_4params.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_strain_fitting(groups, fit_result, save_path, tail_ratio=0.6):
    fig, ax = plt.subplots(figsize=(12, 8))
    color_cycle = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    for i, g in enumerate(groups):
        color = color_cycle[i % len(color_cycle)]
        t_exp = g['t']
        eps_exp = g['strain_rate']
        
        strain_exp = np.zeros_like(t_exp)
        for k in range(1, len(t_exp)):
            dt = t_exp[k] - t_exp[k-1]
            strain_exp[k] = strain_exp[k-1] + 0.5 * (eps_exp[k] + eps_exp[k-1]) * dt
        
        strain_model, rate_model = model_predict_4params(
            t_exp, g['stress'], 
            fit_result['mu'], fit_result['p'], fit_result['beta'], fit_result['a'],
            solver_t_step=0.01
        )
        strain_mod = strain_model - 1.0
        
        mask_se = strain_exp > 0
        ax.plot(t_exp[mask_se], strain_exp[mask_se], 'o',
                color=color, markersize=8, alpha=0.6,
                label=f'Exp $\\sigma$={g["stress"]:.1e}')
        mask_sm = strain_mod > 0
        ax.plot(t_exp[mask_sm], strain_mod[mask_sm], '-',
                color=color, linewidth=3,
                label='Model')
        
        split_idx = int(len(t_exp) * (1.0 - tail_ratio))
        if split_idx < len(t_exp):
            ax.axvline(t_exp[split_idx], color=color, linestyle=':', alpha=0.5)

    ax.set_xlabel('Physical Time $t$', fontsize=label_fontsize)
    ax.set_ylabel('Strain $\\lambda-1$', fontsize=label_fontsize)
    ax.set_xlim(0.0, 10.0)
    ax.set_title(f'Strain Fit (T-Correction Model, $a={fit_result["a"]:.4f}$)', fontsize=title_fontsize, pad=20)
    ax.legend(fontsize=legend_fontsize*0.8, loc='best', ncol=1)
    ax.grid(True, alpha=0.3)
    
    filepath = os.path.join(save_path, "strain_fitting_4params.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_parameter_comparison(fit_result, save_path, true_params=None):
    if true_params is None:
        return
    
    params = ['μ', 'p', 'β', 'a']
    labels = ['$\\mu$', '$p$', '$\\beta$', '$a$']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(params))
    width = 0.35
    
    true_vals = [true_params['mu'], true_params['p'], true_params['beta'], true_params['a']]
    fit_vals = [fit_result['mu'], fit_result['p'], fit_result['beta'], fit_result['a']]
    
    bars1 = ax.bar(x - width/2, true_vals, width, label='True', 
                   color='#2ca02c', alpha=0.8)
    bars2 = ax.bar(x + width/2, fit_vals, width, label='Fitted', 
                   color='#d62728', alpha=0.8)
    
    ax.set_ylabel('Parameter Value', fontsize=label_fontsize)
    ax.set_title('4-Parameter Comparison: True vs Fitted (T-Correction)', fontsize=title_fontsize, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=20)
    
    plt.tight_layout()
    filepath = os.path.join(save_path, "parameter_comparison_4params.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ====================================================================
#  主程序
# ====================================================================

def main():
    print("=" * 60)
    print("2D聚合物网络参数拟合器 (T-Correction 无量纲模型，带迭代计数)")
    print("=" * 60)

    # ================= 【参数配置区 - 都在 main() 内】 =================
    fit_ratio = 0.4          # 拟合后 50% 数据
    solver_t_step = 0.005     # 基础无量纲时间步长
    
    # 初始猜测点
    mu0 = 10.0
    p0 = 2.0
    beta0 = 1.5
    a0 = 0.3

    # 优化搜索边界 (bounds)
    param_bounds = [
        (1e-4, 100.0),     # μ: 弹性模量
        (1.1, 2.0),       # p: 本构指数 (下限提高到1.5，避开 p=1 的非物理区)
        (1e-8, 10.0),      # β: 物理时间缩放因子 (大幅放宽上限，利用动态步长保证速度)
        (0.0, 1.0)         # a: 应变相关修正系数
    ]

    # 3个不同的初始优化出发点
    start_points = [
        [mu0, p0, beta0, a0],
        [mu0*0.1, p0*0.8, beta0*2.0, a0*0.2],
        [mu0*2.0, p0*1.2, beta0*0.5, a0*0.8]
    ]
    # ====================================================================

    data_path = os.path.join(save_path, '4d_creep_BH.xlsx')
    
    if not os.path.exists(data_path):
        print(f"\n实验数据文件不存在: {data_path}")
        print("请确保放入实验数据或生成合成数据用于测试...")
        return
    else:
        true_params = None
        print(f"\n加载实验数据: {data_path}")
    
    groups = load_multi_group_data(data_path)
    print(f"成功加载 {len(groups)} 组数据")
    
    for i, g in enumerate(groups):
        print(f"  组 {i+1}: σ = {g['stress']:.4e}, "
              f"应变率范围 = [{g['strain_rate'].min():.2e}, {g['strain_rate'].max():.2e}]")
    
    # ----------------------- 2. 参数拟合 (传入配置) -----------------------
    fit_res = fit_global_parameters_4params(
        groups, 
        solver_t_step=solver_t_step, 
        tail_ratio=fit_ratio,
        mu_init=mu0, 
        p_init=p0, 
        beta_init=beta0,
        a_init=a0,
        bounds=param_bounds,
        start_points=start_points
    )
    
    if not fit_res:
        print("拟合失败，退出程序")
        return
    
    # ----------------------- 3. 保存拟合参数 -----------------------
    params_path = os.path.join(save_path, "fitted_params_TCorrection.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write(f"# T-Correction 无量纲模型拟合参数\n")
        f.write(f"# Global 4-Parameter Fit (μ, p, β, a)\n")
        f.write(f"# Fitting strategy: Last {int(fit_ratio*100)}% of data\n\n")
        f.write(f"mu   = {fit_res['mu']:.8e}    # Elastic modulus\n")
        f.write(f"p    = {fit_res['p']:.8e}    # Constitutive exponent\n")
        f.write(f"beta = {fit_res['beta']:.8e}    # Time Scaling factor (Phys = Dimless * 1/beta)\n")
        f.write(f"a    = {fit_res['a']:.8e}    # Strain-dependence correction factor (0<=a<=1)\n")
        f.write(f"loss = {fit_res['loss']:.8e}\n")
        f.write(f"success = {fit_res['success']}\n")
    
    if true_params:
        with open(params_path, 'a', encoding='utf-8') as f:
            f.write(f"\n# 真实参数（合成数据）\n")
            f.write(f"mu_true   = {true_params['mu']:.8e}\n")
            f.write(f"p_true    = {true_params['p']:.8e}\n")
            f.write(f"beta_true = {true_params['beta']:.8e}\n")
            f.write(f"a_true    = {true_params['a']:.8e}\n")
    
    print(f"\n拟合参数已保存: {params_path}")
    
    # ----------------------- 4. 可视化拟合结果 -----------------------
    print("\n生成拟合结果可视化...")
    plot_strain_rate_fitting(groups, fit_res, save_path, fit_ratio)
    plot_strain_fitting(groups, fit_res, save_path, fit_ratio)
    
    if true_params:
        plot_parameter_comparison(fit_res, save_path, true_params)
    
    # ----------------------- 5. 生成预测数据集 -----------------------
    print("\n生成模型预测数据...")
    df_pred = pd.DataFrame()
    
    for i, g in enumerate(groups):
        strain_model, rate_model = model_predict_4params(
            g['t'], g['stress'], 
            fit_res['mu'], fit_res['p'], fit_res['beta'], fit_res['a'],
            solver_t_step=0.01
        )
        
        df_pred[f't_group{i+1}'] = g['t']
        df_pred[f'rate_exp_group{i+1}'] = g['strain_rate']
        df_pred[f'rate_model_group{i+1}'] = rate_model
        df_pred[f'strain_exp_group{i+1}'] = np.cumsum(g['strain_rate']) * (g['t'][1]-g['t'][0])
        df_pred[f'strain_model_group{i+1}'] = strain_model - 1.0
    
    pred_path = os.path.join(save_path, "model_predictions_TCorrection.csv")
    df_pred.to_csv(pred_path, index=False, float_format='%.8e')
    print(f"预测数据已保存: {pred_path}")
    
    print("\n" + "=" * 60)
    print("T-Correction 无量纲模型拟合完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()