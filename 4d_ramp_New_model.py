"""
二维平滑 S 型弹性模型 + Meng–Terentjev Transient Network
========================================================

目的
----
在二维不可压缩体系中，考虑动态交联/交换的 transient network，
并使用一个无有限伸长奇点的 phenomenological elastic free energy：

    X = I1 - 2 = lambda^2 + lambda^(-2) - 2

    W_p/mu = 1/2 * [
        X
        - A * (X - Xs*ln(1 + X/Xs))
        + B/(n+1) * (X/Xh)^n * X
    ]

其对应的二维弹性名义应力为：

    sigma_p/mu = (lambda - lambda^(-3)) * H(X)

其中

    H(X) = 1 - A*X/(X+Xs) + B*(X/Xh)^n

该模型不是经典命名的标准超弹性模型，而是针对当前数据形状
构造的 phenomenological ansatz：
    - 第一项：基准橡胶弹性
    - 第二项：中等应变 softening，且逐渐饱和
    - 第三项：大应变 smooth hardening，无 Gent 奇点

Transient network:
------------------
    F(t) = exp(-beta*t) W_p(lambda(t))
         + integral_0^t beta*exp[-beta(t-t')]
           W_p(lambda(t)/lambda(t')) dt'

加载历史：
    lambda(t) = exp(gamma_dot*t)

无量纲化：
    tau = beta*t
    q   = gamma_dot/beta
    lambda = exp(q*tau)

因此：
    a = beta/gamma_dot = 1/q

对于该加载历史，可以将 constitutive relation 写成：

    sigma/mu =
        lambda^(-a) * [sigma_p(lambda)/mu]
        + a/lambda * integral_1^lambda
          Lambda^(-a) [sigma_p(Lambda)/mu] dLambda

程序使用 u = ln(lambda) 做数值积分：
    dLambda = exp(u) du

从而避免直接在 lambda 空间进行积分。

注意
----
1. 这里的 beta 是 single exchange-rate 模型中的交换速率。
2. 不包含 permanent-network fraction nu。
3. sigma/mu 是无量纲应力；mu 仅作为应力尺度。
4. A, Xs, B, Xh, n 为 phenomenological parameters，需要根据数据调整。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from scipy.integrate import cumulative_trapezoid


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
save_path = "/home/tyt/project/Black_hole/phenomenological_transient_results"
os.makedirs(save_path, exist_ok=True)


# ====================================================================
# 1. Phenomenological elastic model
# ====================================================================

def invariant_X(lam):
    """
    二维不可压缩体系：

        I1 = lambda^2 + lambda^(-2)
        X  = I1 - 2
    """
    return lam**2 + lam**(-2) - 2.0


def elastic_energy_hat(lam, A, Xs, B, Xh, n):
    """
    无量纲弹性自由能：

        W_p / mu
        = 1/2 * [
            X
            - A*(X - Xs*ln(1 + X/Xs))
            + B/(n+1)*(X/Xh)^n * X
          ]

    lambda=1 时 W=0。
    """
    lam = np.asarray(lam, dtype=np.float64)
    X = invariant_X(lam)

    if Xs <= 0 or Xh <= 0 or n <= 0:
        raise ValueError("Xs, Xh 和 n 必须为正。")

    softening = X - Xs * np.log1p(X / Xs)
    hardening = (X / Xh)**n * X / (n + 1.0)

    return 0.5 * (X - A * softening + B * hardening)


def elastic_stress_hat(lam, A, Xs, B, Xh, n):
    """
    二维弹性名义应力：

        sigma_p / mu
        = (lambda - lambda^(-3))
          * [1 - A*X/(X+Xs) + B*(X/Xh)^n]
    """
    lam = np.asarray(lam, dtype=np.float64)
    X = invariant_X(lam)

    if Xs <= 0 or Xh <= 0 or n <= 0:
        raise ValueError("Xs, Xh 和 n 必须为正。")

    H = (
        1.0
        - A * X / (X + Xs)
        + B * (X / Xh)**n
    )

    return (lam - lam**(-3.0)) * H


def check_elastic_model(lam, A, Xs, B, Xh, n):
    """
    检查当前参数在指定 lambda 范围内是否保持：

        sigma_p >= 0

    同时给出 H(X) 的最小值。
    """
    X = invariant_X(lam)

    H = (
        1.0
        - A * X / (X + Xs)
        + B * (X / Xh)**n
    )

    sigma_hat = (lam - lam**(-3.0)) * H

    return {
        'H_min': np.min(H),
        'sigma_min': np.min(sigma_hat),
        'sigma_max': np.max(sigma_hat),
        'valid_positive_stress': bool(np.min(sigma_hat) >= -1e-12)
    }


# ====================================================================
# 2. Transient-network constitutive solver
# ====================================================================

def transient_constitutive_curve(
        q,
        lambda_max=3.0,
        n_points=4000,
        A=0.75,
        Xs=0.8,
        B=0.015,
        Xh=1.8,
        n=1.5):
    """
    计算：

        q = gamma_dot / beta

    在

        tau = beta*t

    下的本构曲线。

    加载历史：

        lambda(tau) = exp(q*tau)

    时间范围自动取到 lambda=lambda_max：

        tau_max = ln(lambda_max)/q

    返回：
        tau
        lambda
        sigma_total / mu
        sigma_surviving / mu
        sigma_recrosslinked / mu
        sigma_elastic / mu
    """
    if q <= 0:
        raise ValueError("q = gamma_dot/beta 必须 > 0。")

    if lambda_max <= 1.0:
        raise ValueError("lambda_max 必须 > 1。")

    # u = ln(lambda)
    u_max = np.log(lambda_max)
    u = np.linspace(0.0, u_max, n_points)

    # lambda = exp(u) = exp(q*tau)
    lam = np.exp(u)
    tau = u / q

    # beta/gamma_dot
    a = 1.0 / q

    # 永久橡胶弹性本构
    s_elastic = elastic_stress_hat(
        lam, A, Xs, B, Xh, n
    )

    # ---------------------------------------------------------------
    # 完整 transient constitutive relation
    #
    # sigma_hat =
    #   exp(-a*u) s_p
    #   + a*exp(-u) * integral_0^u
    #       exp[(1-a)v] s_p(exp(v)) dv
    # ---------------------------------------------------------------

    # surviving chains
    sigma_surviving = np.exp(-a * u) * s_elastic

    # re-crosslinked chains
    integrand = np.exp((1.0 - a) * u) * s_elastic
    integral_term = cumulative_trapezoid(
        integrand,
        u,
        initial=0.0
    )

    sigma_recrosslinked = (
        a * np.exp(-u) * integral_term
    )

    sigma_total = (
        sigma_surviving
        + sigma_recrosslinked
    )

    return (
        tau,
        lam,
        sigma_total,
        sigma_surviving,
        sigma_recrosslinked,
        s_elastic
    )


# ====================================================================
# 3. 可视化：总本构曲线
# ====================================================================

def plot_constitutive_curves(
        q_list,
        lambda_max,
        model_params,
        save_path):
    """
    绘制 sigma/mu - lambda。
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    color_cycle = [
        '#d62728',
        '#1f77b4',
        '#2ca02c',
        '#ff7f0e',
        '#9467bd',
        '#8c564b',
        '#e377c2',
    ]

    for i, q in enumerate(q_list):
        color = color_cycle[i % len(color_cycle)]

        (
            tau,
            lam,
            sigma_total,
            sigma_surviving,
            sigma_recrosslinked,
            sigma_elastic
        ) = transient_constitutive_curve(
            q=q,
            lambda_max=lambda_max,
            **model_params
        )

        ax.plot(
            lam,
            sigma_total,
            '-',
            color=color,
            linewidth=4,
            label=rf'$\dot{{\gamma}}/\beta={q:g}$'
        )

    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sigma/\mu$', fontsize=label_fontsize)

    ax.set_xlim(1.0, lambda_max)

    ax.set_title(
        '2D Phenomenological Elasticity + Transient Network',
        fontsize=title_fontsize,
        pad=20
    )

    ax.legend(
        fontsize=legend_fontsize,
        loc='best'
    )

    ax.tick_params(
        axis='both',
        which='major',
        width=2,
        length=10
    )

    ax.tick_params(
        axis='both',
        which='minor',
        width=1.5,
        length=5
    )

    ax.minorticks_on()

    ax.grid(
        True,
        alpha=0.3,
        linewidth=1
    )

    filepath = os.path.join(
        save_path,
        'phenomenological_transient_constitutive_curves.png'
    )

    fig.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close()

    print(f"本构曲线已保存: {filepath}")


# ====================================================================
# 4. 可视化：应力组成
# ====================================================================

def plot_stress_components(
        q,
        lambda_max,
        model_params,
        save_path):
    """
    对指定 q，分别显示：

        1. surviving contribution
        2. re-crosslinked contribution
        3. total stress
        4. equilibrium elastic stress
    """
    (
        tau,
        lam,
        sigma_total,
        sigma_surviving,
        sigma_recrosslinked,
        sigma_elastic
    ) = transient_constitutive_curve(
        q=q,
        lambda_max=lambda_max,
        **model_params
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        lam,
        sigma_total,
        '-',
        color='black',
        linewidth=4,
        label='Total transient stress'
    )

    ax.plot(
        lam,
        sigma_surviving,
        '--',
        linewidth=3,
        label='Surviving chains'
    )

    ax.plot(
        lam,
        sigma_recrosslinked,
        ':',
        linewidth=4,
        label='Re-crosslinked chains'
    )

    ax.plot(
        lam,
        sigma_elastic,
        '-.',
        linewidth=3,
        label='Elastic reference curve'
    )

    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sigma/\mu$', fontsize=label_fontsize)

    ax.set_xlim(1.0, lambda_max)

    ax.set_title(
        rf'Stress Components, $\dot{{\gamma}}/\beta={q:g}$',
        fontsize=title_fontsize,
        pad=20
    )

    ax.legend(
        fontsize=legend_fontsize * 0.85,
        loc='best'
    )

    ax.tick_params(
        axis='both',
        which='major',
        width=2,
        length=10
    )

    ax.minorticks_on()

    ax.grid(
        True,
        alpha=0.3,
        linewidth=1
    )

    filepath = os.path.join(
        save_path,
        f'stress_components_q_{q:g}.png'
    )

    fig.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close()

    print(f"应力组成图已保存: {filepath}")


# ====================================================================
# 5. 保存无量纲数据
# ====================================================================

def save_prediction_data(
        q_list,
        lambda_max,
        model_params,
        save_path):
    """
    每个 q 单独保存一个 CSV。
    """
    for q in q_list:
        (
            tau,
            lam,
            sigma_total,
            sigma_surviving,
            sigma_recrosslinked,
            sigma_elastic
        ) = transient_constitutive_curve(
            q=q,
            lambda_max=lambda_max,
            **model_params
        )

        data = np.column_stack([
            tau,
            lam,
            sigma_total,
            sigma_surviving,
            sigma_recrosslinked,
            sigma_elastic
        ])

        filepath = os.path.join(
            save_path,
            f'prediction_q_{q:g}.csv'
        )

        header = (
            'tau=beta*t,lambda,'
            'sigma_total_over_mu,'
            'sigma_surviving_over_mu,'
            'sigma_recrosslinked_over_mu,'
            'sigma_elastic_over_mu'
        )

        np.savetxt(
            filepath,
            data,
            delimiter=',',
            header=header,
            comments='',
            fmt='%.10e'
        )

        print(f"预测数据已保存: {filepath}")


# ====================================================================
# 6. 主程序
# ====================================================================

def main():

    print("=" * 70)
    print("二维平滑 S 型弹性模型 + Transient Network")
    print("加载历史: lambda(t) = exp(gamma_dot*t)")
    print("时间单位: 1/beta")
    print("应力单位: mu")
    print("=" * 70)

    # =================================================================
    # 参数配置区
    # =================================================================

    # ---------- Dynamic parameters ----------
    # q = gamma_dot / beta
    q_list = [0.1, 0.2, 0.4, 0.6, 0.8]

    # ---------- Stretch range ----------
    lambda_max = 10.0

    # ---------- Numerical resolution ----------
    n_points = 4000

    # ---------- Phenomenological elastic parameters ----------
    #
    # sigma_p/mu =
    # (lambda-lambda^-3)
    # [1 - A*X/(X+Xs) + B*(X/Xh)^n]
    #
    # 注意：这些是示例参数，不是实验拟合值。
    #
    A = 0.75
    Xs = 0.80

    B = 0.015
    Xh = 1.80
    n = 1.50

    model_params = {
        'A': A,
        'Xs': Xs,
        'B': B,
        'Xh': Xh,
        'n': n
    }

    # =================================================================
    # 参数检查
    # =================================================================

    lam_check = np.linspace(
        1.0,
        lambda_max,
        3000
    )

    check = check_elastic_model(
        lam_check,
        **model_params
    )

    print("\n弹性模型检查:")
    print(f"  H(X) 最小值       = {check['H_min']:.6e}")
    print(f"  sigma_p/mu 最小值 = {check['sigma_min']:.6e}")
    print(f"  sigma_p/mu 最大值 = {check['sigma_max']:.6e}")

    if not check['valid_positive_stress']:
        print(
            "\n警告：当前参数在指定 lambda 区间内产生了负的 "
            "elastic stress，请重新调整 A, Xs, B, Xh, n。"
        )

    print("\n模型参数:")
    print(f"  A  = {A:.6f}")
    print(f"  Xs = {Xs:.6f}")
    print(f"  B  = {B:.6f}")
    print(f"  Xh = {Xh:.6f}")
    print(f"  n  = {n:.6f}")

    # =================================================================
    # 生成结果
    # =================================================================

    print("\n生成本构曲线...")
    plot_constitutive_curves(
        q_list=q_list,
        lambda_max=lambda_max,
        model_params=model_params,
        save_path=save_path
    )

    print("\n生成代表性应力分解图...")
    plot_stress_components(
        q=0.6,
        lambda_max=lambda_max,
        model_params=model_params,
        save_path=save_path
    )

    print("\n保存预测数据...")
    save_prediction_data(
        q_list=q_list,
        lambda_max=lambda_max,
        model_params=model_params,
        save_path=save_path
    )

    # =================================================================
    # 保存参数
    # =================================================================

    params_file = os.path.join(
        save_path,
        'model_parameters.txt'
    )

    with open(
        params_file,
        'w',
        encoding='utf-8'
    ) as f:

        f.write(
            '# 2D phenomenological elastic model + transient network\n'
        )
        f.write(
            '# lambda(t) = exp(gamma_dot*t)\n'
        )
        f.write(
            '# tau = beta*t, q = gamma_dot/beta\n\n'
        )

        f.write(
            f'A  = {A:.10e}\n'
        )
        f.write(
            f'Xs = {Xs:.10e}\n'
        )
        f.write(
            f'B  = {B:.10e}\n'
        )
        f.write(
            f'Xh = {Xh:.10e}\n'
        )
        f.write(
            f'n  = {n:.10e}\n'
        )

        f.write(
            f'\nlambda_max = {lambda_max:.10e}\n'
        )
        f.write(
            f'n_points   = {n_points}\n'
        )

        f.write(
            '\nq = gamma_dot/beta:\n'
        )

        for q in q_list:
            f.write(
                f'  {q:.10e}\n'
            )

    print(
        f"\n模型参数已保存: {params_file}"
    )

    print("\n" + "=" * 70)
    print("计算完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
