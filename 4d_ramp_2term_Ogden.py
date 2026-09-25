"""
二维 Two-term Ogden + Transient Network 本构计算

模型框架
--------
1. 二维不可压缩变形：
       Lambda(t, t') = lambda(t) / lambda(t')
       I1 = Lambda^2 + Lambda^(-2)

2. Two-term Ogden 橡胶自由能：
       Wp(Lambda) / mu = sum_i (mu_i/mu)/p_i *
                          [Lambda^p_i + Lambda^(-p_i) - 2]

   其中 mu 为应力归一化单位，mu_i/mu 为无量纲权重。

3. Transient-network 自由能：
       F(t) = exp(-beta t) Wp(lambda(t))
              + integral_0^t beta exp[-beta(t-t')] *
                Wp(lambda(t)/lambda(t')) dt'

4. 指定拉伸历史：
       lambda(t) = exp(gamma_dot * t)

   使用 tau = beta*t 作为无量纲时间：
       tau = beta*t
       q   = gamma_dot / beta
       lambda = exp(q*tau)

5. 在这种加载历史下，本构方程可写为：
       sigma / mu = lambda^(-a) * s_p(lambda)
                    + a/lambda * integral_1^lambda
                      Lambda^(-a) s_p(Lambda) dLambda

   其中：
       a = beta/gamma_dot = 1/q
       s_p = dWp/dlambda / mu

   对每一个 Ogden mode，可得到解析表达式，因此无需数值积分。

注意
----
- 这里是“纯 transient network”模型，不含 permanent-network fraction nu。
- 本程序针对用户当前问题采用二维不变量 I1 = lambda^2 + lambda^-2。
- 默认参数仅用于展示模型行为，不代表对实验数据的拟合结果。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
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
markersize = 12

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
save_path = "/home/tyt/project/Black_hole/2term_ogden_transient_results"
os.makedirs(save_path, exist_ok=True)


# ====================================================================
#  Two-term Ogden: 单个 mode 的无量纲瞬态响应
# ====================================================================

def _int_power_over_x(power, x):
    """
    计算 integral_1^lambda Lambda^(power-1) dLambda

    即：
        (lambda^power - 1) / power, power != 0
        ln(lambda),                       power == 0

    使用 expm1 + log 提高接近 power = 0 时的数值稳定性。
    """
    x = np.asarray(x, dtype=np.float64)
    log_x = np.log(x)

    if abs(power) < 1e-10:
        return log_x

    return np.expm1(power * log_x) / power


def ogden_mode_transient(lambda_val, a, p):
    """
    计算单个 Ogden mode 对 sigma/mu 的贡献。

    elastic stress of one mode:
        s_p(lambda)/mu = lambda^(p-1) - lambda^(-p-1)

    transient response:
        phi_p(lambda,a) = lambda^(-a) s_p(lambda)/mu
                         + a/lambda * integral_1^lambda
                           Lambda^(-a) s_p(Lambda)/mu dLambda

    参数
    ----
    lambda_val : array_like
        Stretch ratio lambda >= 1.
    a : float
        beta / gamma_dot.
    p : float
        Ogden exponent.
    """
    lam = np.asarray(lambda_val, dtype=np.float64)

    if np.any(lam < 1.0):
        raise ValueError("本程序当前针对单轴拉伸，要求 lambda >= 1。")
    if a <= 0.0:
        raise ValueError("a = beta/gamma_dot 必须大于 0。")
    if p <= 0.0:
        raise ValueError("当前代码要求 Ogden exponent p > 0。")

    # ---------------------------------------------------------------
    # 第一部分：最初形成且到当前时刻仍然存活的网络链
    # ---------------------------------------------------------------
    elastic_stress = lam ** (p - 1.0) - lam ** (-p - 1.0)
    survival_part = lam ** (-a) * elastic_stress

    # ---------------------------------------------------------------
    # 第二部分：在 0 < t' < t 之间重新形成的网络链
    # ---------------------------------------------------------------
    # Integral Lambda^(-a) * [Lambda^(p-1) - Lambda^(-p-1)] dLambda
    # = Integral Lambda^(p-a-1) dLambda
    #   - Integral Lambda^(-p-a-1) dLambda
    # 第二项：- [ (lambda^(-p-a)-1) / (-p-a) ]
    #       = + [ (lambda^(-p-a)-1) / (p+a) ]

    first_integral = _int_power_over_x(p - a, lam)
    second_integral = _int_power_over_x(-(p + a), lam)

    reborn_part = (a / lam) * (first_integral + second_integral)

    return survival_part + reborn_part


# ====================================================================
#  Two-term Ogden 总本构
# ====================================================================

def two_term_ogden_stress(lambda_val, gamma_dot_over_beta,
                           mu1_over_mu, mu2_over_mu,
                           p1, p2):
    """
    计算二维 Two-term Ogden + transient network 的无量纲应力 sigma/mu。

    输入的 gamma_dot_over_beta 定义为：
        q = gamma_dot / beta

    拉伸历史：
        lambda(tau) = exp(q * tau)

    返回：
        sigma / mu
    """
    q = float(gamma_dot_over_beta)

    if q <= 0.0:
        raise ValueError("gamma_dot/beta 必须大于 0。")

    a = 1.0 / q

    phi1 = ogden_mode_transient(lambda_val, a, p1)
    phi2 = ogden_mode_transient(lambda_val, a, p2)

    return mu1_over_mu * phi1 + mu2_over_mu * phi2


# ====================================================================
#  生成指定 q 下的 lambda(t), tau 和 sigma/mu
# ====================================================================

def compute_curve(q, lambda_max=3.0, n_points=1200,
                  mu1_over_mu=0.98, mu2_over_mu=0.02,
                  p1=1.10, p2=4.50):
    """
    生成一条本构曲线。

    时间使用 tau = beta*t；
    应变历史使用 lambda = exp(q*tau)，其中 q = gamma_dot/beta。
    """
    if lambda_max <= 1.0:
        raise ValueError("lambda_max 必须 > 1。")
    if n_points < 10:
        raise ValueError("n_points 太小。")

    # tau_max 由 lambda_max = exp(q*tau_max) 得到
    tau_max = np.log(lambda_max) / q

    tau = np.linspace(0.0, tau_max, n_points)
    lam = np.exp(q * tau)

    sigma_norm = two_term_ogden_stress(
        lam,
        gamma_dot_over_beta=q,
        mu1_over_mu=mu1_over_mu,
        mu2_over_mu=mu2_over_mu,
        p1=p1,
        p2=p2,
    )

    return tau, lam, sigma_norm


# ====================================================================
#  可视化 1：不同拉伸速率下的本构曲线 sigma/mu - lambda
# ====================================================================

def plot_constitutive_curves(q_list, lambda_max,
                             mu1_over_mu, mu2_over_mu,
                             p1, p2):
    """
    绘制 sigma/mu - lambda。
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    color_cycle = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for i, q in enumerate(q_list):
        tau, lam, sigma_norm = compute_curve(
            q=q,
            lambda_max=lambda_max,
            n_points=1200,
            mu1_over_mu=mu1_over_mu,
            mu2_over_mu=mu2_over_mu,
            p1=p1,
            p2=p2,
        )

        color = color_cycle[i % len(color_cycle)]

        ax.plot(
            lam,
            sigma_norm,
            '-',
            color=color,
            linewidth=lines_linewidth,
            label=rf'$\dot{{\gamma}}/\beta={q:g}$'
        )

    ax.set_xlim(1.0, lambda_max)
    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sigma/\mu$', fontsize=label_fontsize)

    ax.set_title(
        '2D Two-term Ogden + Transient Network',
        fontsize=title_fontsize,
        pad=20,
    )

    ax.legend(fontsize=legend_fontsize * 0.9, loc='best')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(save_path, 'two_term_ogden_constitutive_curves.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"本构曲线已保存: {filepath}")


# ====================================================================
#  可视化 2：示意性展示“存活链 + 新生链 + 总响应”
# ====================================================================

def plot_response_components(q,
                             lambda_max,
                             mu1_over_mu, mu2_over_mu,
                             p1, p2):
    """
    对指定 q，绘制：
        survival contribution
        reborn contribution
        total stress

    用于判断中段 softening / 后段 hardening 的来源。
    """
    tau, lam, _ = compute_curve(
        q=q,
        lambda_max=lambda_max,
        n_points=1200,
        mu1_over_mu=mu1_over_mu,
        mu2_over_mu=mu2_over_mu,
        p1=p1,
        p2=p2,
    )

    a = 1.0 / q

    phi1_survival = lam ** (-a) * (
        lam ** (p1 - 1.0) - lam ** (-p1 - 1.0)
    )
    phi2_survival = lam ** (-a) * (
        lam ** (p2 - 1.0) - lam ** (-p2 - 1.0)
    )

    survival = (
        mu1_over_mu * phi1_survival
        + mu2_over_mu * phi2_survival
    )

    total = two_term_ogden_stress(
        lam,
        gamma_dot_over_beta=q,
        mu1_over_mu=mu1_over_mu,
        mu2_over_mu=mu2_over_mu,
        p1=p1,
        p2=p2,
    )

    reborn = total - survival

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(lam, survival, '-', linewidth=lines_linewidth,
            label='Surviving chains')
    ax.plot(lam, reborn, '--', linewidth=lines_linewidth,
            label='Re-crosslinked chains')
    ax.plot(lam, total, '-.', linewidth=lines_linewidth,
            label='Total stress')

    ax.set_xlim(1.0, lambda_max)
    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sigma/\mu$', fontsize=label_fontsize)
    ax.set_title(
        rf'Contributions at $\dot{{\gamma}}/\beta={q:g}$',
        fontsize=title_fontsize,
        pad=20,
    )
    ax.legend(fontsize=legend_fontsize * 0.9, loc='best')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(save_path, 'two_term_ogden_components.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"响应分量图已保存: {filepath}")


# ====================================================================
#  输出示例数据
# ====================================================================

def save_example_data(q_list, lambda_max,
                      mu1_over_mu, mu2_over_mu,
                      p1, p2):
    """
    保存每一个 q 对应的 tau, lambda, sigma/mu。
    """
    for q in q_list:
        tau, lam, sigma_norm = compute_curve(
            q=q,
            lambda_max=lambda_max,
            n_points=1200,
            mu1_over_mu=mu1_over_mu,
            mu2_over_mu=mu2_over_mu,
            p1=p1,
            p2=p2,
        )

        data = np.column_stack((tau, lam, sigma_norm))

        filepath = os.path.join(
            save_path,
            f'two_term_ogden_q_{q:g}.txt'
        )

        header = (
            'tau=beta*t, lambda=exp[(gamma_dot/beta)*tau], '
            'sigma_norm=sigma/mu\n'
            'tau lambda sigma_over_mu'
        )

        np.savetxt(filepath, data, header=header)
        print(f"预测数据已保存: {filepath}")


# ====================================================================
#  主程序
# ====================================================================

def main():
    """
    主程序：
    计算二维 Two-term Ogden + transient network 本构曲线。
    """
    print("=" * 70)
    print("2D Two-term Ogden + Transient Network Constitutive Model")
    print("=" * 70)

    # ================= 【参数配置区】 =================

    # ------------------ 动力学参数 ------------------
    # q = gamma_dot / beta
    # tau = beta*t
    q_list = [0.1, 0.2, 0.4, 0.6, 0.8]

    # ------------------ 变形范围 ------------------
    lambda_max = 3.0

    # ------------------ Two-term Ogden ------------------
    # sigma 以 mu 为单位，因此只输入无量纲 mu_i/mu
    # 不要求 mu1/mu + mu2/mu = 1，可自由调整
    mu1_over_mu = 0.98
    mu2_over_mu = 0.02

    # 两个 Ogden exponent
    p1 = 1.10
    p2 = 4.50

    # ====================================================

    print("\n模型参数:")
    print(f"  mu1/mu = {mu1_over_mu:.6g}")
    print(f"  mu2/mu = {mu2_over_mu:.6g}")
    print(f"  p1     = {p1:.6g}")
    print(f"  p2     = {p2:.6g}")
    print(f"  lambda_max = {lambda_max:.6g}")
    print(f"  q=gamma_dot/beta = {q_list}")

    print("\n生成本构曲线...")
    plot_constitutive_curves(
        q_list=q_list,
        lambda_max=lambda_max,
        mu1_over_mu=mu1_over_mu,
        mu2_over_mu=mu2_over_mu,
        p1=p1,
        p2=p2,
    )

    # 选一个代表性 q，观察各动力学贡献
    q_component = 0.5
    print("\n生成响应分量图...")
    plot_response_components(
        q=q_component,
        lambda_max=lambda_max,
        mu1_over_mu=mu1_over_mu,
        mu2_over_mu=mu2_over_mu,
        p1=p1,
        p2=p2,
    )

    print("\n保存预测数据...")
    save_example_data(
        q_list=q_list,
        lambda_max=lambda_max,
        mu1_over_mu=mu1_over_mu,
        mu2_over_mu=mu2_over_mu,
        p1=p1,
        p2=p2,
    )

    print("\n" + "=" * 70)
    print("计算完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()