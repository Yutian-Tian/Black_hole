"""
2D Dynamic-State Elasticity + Transient Network
================================================

模型目标
--------
在不改变 bond-exchange rate beta 的前提下，引入一个内部网络状态变量 xi，
描述 polymer strand 的构象/取向/内部松弛。

核心思想：
    beta : 拓扑更新（bond exchange）的时间尺度
    tau_xi: 链内部构象松弛的时间尺度

加载：
    lambda(t) = exp(gamma_dot * t)

无量纲：
    tau = beta*t
    q   = gamma_dot/beta
    chi = beta*tau_xi

因此：
    lambda(tau) = exp(q*tau)

内部变量：
    dxi/dtau = [xi_eq(lambda) - xi]/chi

二维构象各向异性的一个自然候选：
    xi_eq(lambda)
      = (lambda^2 - lambda^-2)
        /(lambda^2 + lambda^-2)

弹性模型
--------
采用最简单的“动态二阶 Yeoh”形式：

    X = Lambda^2 + Lambda^-2 - 2

    W_p/mu = 1/2 * [X + c2(xi)*X^2]

    c2(xi) = c20 + c21*xi

对应二维 nominal stress：

    s_p/mu
      = (Lambda - Lambda^-3)
        * [1 + 2*c2(xi)*X]

Transient network
-----------------
保留 constant-beta 的 Meng–Terentjev dynamic-reference-state 框架：

    sigma/mu =
        exp(-tau) * s_p(lambda(t), xi(t))/mu

      + integral_0^tau
          exp[-(tau-u)]
          * s_p(lambda(t)/lambda(u), xi(t))/lambda(u)
        du

注意：这里 xi(t) 是一个 mean-field 内部状态变量；
它不为每一个 birth time u 再引入独立的 xi(u)，从而避免
模型复杂度变成双时间内部变量问题。

程序输出
--------
1. 总本构曲线 sigma/mu - lambda
2. xi(lambda) - lambda
3. 指定 q 下的应力分解：
       surviving chains
       re-crosslinked chains
       total stress
       instantaneous elastic stress
4. CSV 预测数据
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os


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
save_path = "/home/tyt/project/Black_hole/dynamic_xi_transient_results"
os.makedirs(save_path, exist_ok=True)


# ====================================================================
# 1. 二维内部状态变量
# ====================================================================

def xi_equilibrium(lam):
    """
    当前二维 affine deformation 对应的平衡构象各向异性：

        xi_eq =
        (lambda^2 - lambda^-2)
        /(lambda^2 + lambda^-2)

    满足：
        xi_eq(1) = 0
        xi_eq(lambda -> infinity) -> 1
    """
    lam = np.asarray(lam, dtype=np.float64)

    numerator = lam**2 - lam**(-2.0)
    denominator = lam**2 + lam**(-2.0)

    return numerator / denominator


def solve_xi(q, chi, lambda_max=3.0, n_points=1800):
    """
    在 tau=beta*t 下求解：

        dxi/dtau = [xi_eq(lambda)-xi]/chi

        lambda = exp(q*tau)

    使用 RK4。
    """
    if q <= 0:
        raise ValueError("q = gamma_dot/beta 必须 > 0。")

    if chi <= 0:
        raise ValueError("chi = beta*tau_xi 必须 > 0。")

    tau_max = np.log(lambda_max) / q

    tau = np.linspace(
        0.0,
        tau_max,
        n_points
    )

    lam = np.exp(q * tau)

    xi = np.zeros_like(tau)

    def rhs(tau_local, xi_local):
        lam_local = np.exp(q * tau_local)
        xi_eq = xi_equilibrium(lam_local)
        return (xi_eq - xi_local) / chi

    for i in range(1, n_points):

        dt = tau[i] - tau[i - 1]

        t0 = tau[i - 1]
        x0 = xi[i - 1]

        k1 = rhs(t0, x0)
        k2 = rhs(t0 + 0.5 * dt, x0 + 0.5 * dt * k1)
        k3 = rhs(t0 + 0.5 * dt, x0 + 0.5 * dt * k2)
        k4 = rhs(t0 + dt, x0 + dt * k3)

        xi[i] = x0 + dt * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0

    return tau, lam, xi


# ====================================================================
# 2. 弹性模型
# ====================================================================

def invariant_X(lam):
    """
    二维不可压缩：

        X = I1 - 2
          = lambda^2 + lambda^-2 - 2
    """
    lam = np.asarray(lam, dtype=np.float64)

    return lam**2 + lam**(-2.0) - 2.0


def c2_dynamic(xi, c20, c21):
    """
    内部状态相关的 nonlinear-elasticity coefficient：

        c2 = c20 + c21*xi
    """
    return c20 + c21 * xi


def elastic_stress_hat(
        lam,
        xi,
        c20,
        c21):
    """
    动态二阶 Yeoh：

        W/mu = 1/2 [X + c2(xi) X^2]

        sigma_p/mu
          = (lambda-lambda^-3)
            [1 + 2*c2(xi)*X]
    """
    lam = np.asarray(lam, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)

    X = invariant_X(lam)

    c2 = c2_dynamic(
        xi,
        c20,
        c21
    )

    return (
        lam - lam**(-3.0)
    ) * (
        1.0 + 2.0 * c2 * X
    )


def elastic_energy_hat(
        lam,
        xi,
        c20,
        c21):
    """
    W_p/mu。
    """
    X = invariant_X(lam)

    c2 = c2_dynamic(
        xi,
        c20,
        c21
    )

    return 0.5 * (
        X + c2 * X**2
    )


# ====================================================================
# 3. Transient-network constitutive equation
# ====================================================================

def compute_transient_curve(
        q,
        chi,
        c20,
        c21,
        lambda_max=3.0,
        n_points=1800):
    """
    计算给定 q=gamma_dot/beta、chi=beta*tau_xi 下的
    transient-network 本构响应。

    tau = beta*t
    lambda = exp(q*tau)

    总应力：

        sigma_hat(tau)
        = exp(-tau) * s_p(lambda(t),xi(t))

        + integral_0^tau
            exp[-(tau-u)]
            s_p(exp[q(tau-u)],xi(t))
            / exp(q*u) du

    """
    (
        tau,
        lam,
        xi
    ) = solve_xi(
        q=q,
        chi=chi,
        lambda_max=lambda_max,
        n_points=n_points
    )

    # instantaneous elastic stress
    sigma_elastic = elastic_stress_hat(
        lam,
        xi,
        c20,
        c21
    )

    # surviving initial network
    sigma_surviving = (
        np.exp(-tau)
        * sigma_elastic
    )

    # re-crosslinked contribution
    sigma_recrosslinked = np.zeros_like(tau)

    for i in range(1, n_points):

        u = tau[:i + 1]

        # s = tau_i - u
        s = tau[i] - u

        # current cohort deformation ratio
        Lambda = np.exp(q * s)

        sigma_cohort = elastic_stress_hat(
            Lambda,
            xi[i],
            c20,
            c21
        )

        # 1/lambda(u) factor is essential for the 2D
        # uniaxial nominal-stress definition used here.
        integrand = (
            np.exp(-s)
            * sigma_cohort
            / np.exp(q * u)
        )

        sigma_recrosslinked[i] = np.trapezoid(
            integrand,
            u
        )

    sigma_total = (
        sigma_surviving
        + sigma_recrosslinked
    )

    return {
        'tau': tau,
        'lambda': lam,
        'xi': xi,
        'sigma_total': sigma_total,
        'sigma_surviving': sigma_surviving,
        'sigma_recrosslinked': sigma_recrosslinked,
        'sigma_elastic': sigma_elastic
    }


# ====================================================================
# 4. 主本构曲线
# ====================================================================

def plot_constitutive_curves(
        q_list,
        chi,
        lambda_max,
        n_points,
        model_params):
    """
    sigma/mu - lambda。
    """
    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    color_cycle = [
        '#d62728',
        '#1f77b4',
        '#2ca02c',
        '#ff7f0e',
        '#9467bd',
        '#8c564b',
        '#e377c2'
    ]

    for i, q in enumerate(q_list):

        result = compute_transient_curve(
            q=q,
            chi=chi,
            lambda_max=lambda_max,
            n_points=n_points,
            **model_params
        )

        color = color_cycle[
            i % len(color_cycle)
        ]

        ax.plot(
            result['lambda'],
            result['sigma_total'],
            '-',
            color=color,
            linewidth=4,
            label=rf'$\dot{{\gamma}}/\beta={q:g}$'
        )

    ax.set_xlabel(
        r'$\lambda$',
        fontsize=label_fontsize
    )

    ax.set_ylabel(
        r'$\sigma/\mu$',
        fontsize=label_fontsize
    )

    ax.set_xlim(
        1.0,
        lambda_max
    )

    ax.set_title(
        rf'2D Dynamic-State Elasticity + Transient Network'
        '\n'
        rf'$\chi=\beta\tau_\xi={chi:g}$',
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
        'dynamic_xi_transient_constitutive_curves.png'
    )

    fig.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close()

    print(
        f"本构曲线已保存: {filepath}"
    )


# ====================================================================
# 5. xi(lambda) 可视化
# ====================================================================

def plot_xi_curves(
        q_list,
        chi,
        lambda_max,
        n_points):
    """
    显示不同拉伸速率下内部变量 xi(lambda)。
    """
    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    color_cycle = [
        '#d62728',
        '#1f77b4',
        '#2ca02c',
        '#ff7f0e',
        '#9467bd'
    ]

    for i, q in enumerate(q_list):

        (
            tau,
            lam,
            xi
        ) = solve_xi(
            q=q,
            chi=chi,
            lambda_max=lambda_max,
            n_points=n_points
        )

        color = color_cycle[
            i % len(color_cycle)
        ]

        ax.plot(
            lam,
            xi,
            '-',
            color=color,
            linewidth=4,
            label=rf'$\dot{{\gamma}}/\beta={q:g}$'
        )

    lam_eq = np.linspace(
        1.0,
        lambda_max,
        1000
    )

    ax.plot(
        lam_eq,
        xi_equilibrium(lam_eq),
        '--',
        color='black',
        linewidth=3,
        label=r'$\xi_{\rm eq}(\lambda)$'
    )

    ax.set_xlabel(
        r'$\lambda$',
        fontsize=label_fontsize
    )

    ax.set_ylabel(
        r'$\xi$',
        fontsize=label_fontsize
    )

    ax.set_xlim(
        1.0,
        lambda_max
    )

    ax.set_title(
        rf'Internal Conformational State, $\chi={chi:g}$',
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

    ax.minorticks_on()

    ax.grid(
        True,
        alpha=0.3,
        linewidth=1
    )

    filepath = os.path.join(
        save_path,
        'xi_vs_lambda.png'
    )

    fig.savefig(
        filepath,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.close()

    print(
        f"xi 曲线已保存: {filepath}"
    )


# ====================================================================
# 6. 应力分解
# ====================================================================

def plot_stress_components(
        q,
        chi,
        lambda_max,
        n_points,
        model_params):
    """
    显示指定 q 下的应力组成。
    """
    result = compute_transient_curve(
        q=q,
        chi=chi,
        lambda_max=lambda_max,
        n_points=n_points,
        **model_params
    )

    lam = result['lambda']

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    ax.plot(
        lam,
        result['sigma_total'],
        '-',
        color='black',
        linewidth=4,
        label='Total'
    )

    ax.plot(
        lam,
        result['sigma_surviving'],
        '--',
        linewidth=3,
        label='Surviving'
    )

    ax.plot(
        lam,
        result['sigma_recrosslinked'],
        ':',
        linewidth=4,
        label='Re-crosslinked'
    )

    ax.plot(
        lam,
        result['sigma_elastic'],
        '-.',
        linewidth=3,
        label='Instantaneous elastic'
    )

    ax.set_xlabel(
        r'$\lambda$',
        fontsize=label_fontsize
    )

    ax.set_ylabel(
        r'$\sigma/\mu$',
        fontsize=label_fontsize
    )

    ax.set_xlim(
        1.0,
        lambda_max
    )

    ax.set_title(
        rf'Stress Components, '
        rf'$\dot{{\gamma}}/\beta={q:g}$, '
        rf'$\chi={chi:g}$',
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

    print(
        f"应力分解图已保存: {filepath}"
    )


# ====================================================================
# 7. 保存数据
# ====================================================================

def save_prediction_data(
        q_list,
        chi,
        lambda_max,
        n_points,
        model_params):
    """
    每个 q 保存一组 CSV。
    """
    for q in q_list:

        result = compute_transient_curve(
            q=q,
            chi=chi,
            lambda_max=lambda_max,
            n_points=n_points,
            **model_params
        )

        data = np.column_stack([
            result['tau'],
            result['lambda'],
            result['xi'],
            result['sigma_total'],
            result['sigma_surviving'],
            result['sigma_recrosslinked'],
            result['sigma_elastic']
        ])

        filepath = os.path.join(
            save_path,
            f'prediction_q_{q:g}.csv'
        )

        header = (
            'tau=beta*t,'
            'lambda,'
            'xi,'
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

        print(
            f"预测数据已保存: {filepath}"
        )


# ====================================================================
# 8. 主程序
# ====================================================================

def main():

    print("=" * 70)
    print("2D Dynamic-State Elasticity + Transient Network")
    print("lambda(t) = exp(gamma_dot*t)")
    print("tau = beta*t")
    print("q = gamma_dot/beta")
    print("=" * 70)

    # =================================================================
    # 参数配置区
    # =================================================================

    # ---------- 外部加载 ----------
    q_list = [
        0.1,
        0.2,
        0.4,
        0.6,
        0.8
    ]

    # ---------- 内部状态变量时间尺度 ----------
    # chi = beta*tau_xi
    #
    # chi ~ 1:
    #   internal conformational relaxation
    #   与 bond exchange 为同一量级
    #
    chi = 1.0

    # ---------- stretch ----------
    lambda_max = 3.0

    # ---------- numerical resolution ----------
    n_points = 1800

    # ---------- elastic parameters ----------
    #
    # c2(xi) = c20 + c21*xi
    #
    # 当前值仅用于验证模型形状。
    c20 = 0.01
    c21 = 0.02

    model_params = {
        'c20': c20,
        'c21': c21
    }

    print("\n模型参数:")
    print(f"  chi = beta*tau_xi = {chi:.6f}")
    print(f"  c20 = {c20:.6f}")
    print(f"  c21 = {c21:.6f}")

    # =================================================================
    # 计算
    # =================================================================

    print("\n生成本构曲线...")

    plot_constitutive_curves(
        q_list=q_list,
        chi=chi,
        lambda_max=lambda_max,
        n_points=n_points,
        model_params=model_params
    )

    print("\n生成内部状态变量曲线...")

    plot_xi_curves(
        q_list=q_list,
        chi=chi,
        lambda_max=lambda_max,
        n_points=n_points
    )

    print("\n生成应力分解图...")

    plot_stress_components(
        q=0.4,
        chi=chi,
        lambda_max=lambda_max,
        n_points=n_points,
        model_params=model_params
    )

    print("\n保存预测数据...")

    save_prediction_data(
        q_list=q_list,
        chi=chi,
        lambda_max=lambda_max,
        n_points=n_points,
        model_params=model_params
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
            '# 2D Dynamic-State Elasticity + Transient Network\n'
        )
        f.write(
            '# lambda(t) = exp(gamma_dot*t)\n'
        )
        f.write(
            '# tau = beta*t\n'
        )
        f.write(
            '# q = gamma_dot/beta\n'
        )
        f.write(
            '# chi = beta*tau_xi\n\n'
        )

        f.write(
            f'chi = {chi:.10e}\n'
        )

        f.write(
            f'c20 = {c20:.10e}\n'
        )

        f.write(
            f'c21 = {c21:.10e}\n'
        )

        f.write(
            f'lambda_max = {lambda_max:.10e}\n'
        )

        f.write(
            f'n_points = {n_points}\n'
        )

        f.write(
            '\nq = gamma_dot/beta:\n'
        )

        for q in q_list:
            f.write(
                f'  {q:.10e}\n'
            )

    print(
        f"\n参数文件已保存: {params_file}"
    )

    print("\n" + "=" * 70)
    print("计算完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
