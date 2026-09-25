"""
二维五阶 Yeoh（Reduced Polynomial）+ Transient Network 本构计算

模型框架
--------
1. 二维不可压缩变形：
       Lambda(t,t') = lambda(t) / lambda(t')
       I1 = Lambda^2 + Lambda^(-2)
       X  = I1 - 2

2. 五阶 Yeoh / reduced-polynomial 橡胶自由能：
       Wp(Lambda)/mu = 1/2 * [
           X + c2 X^2 + c3 X^3 + c4 X^4 + c5 X^5
       ]

   其中 mu 是应力归一化尺度，c2~c5 为无量纲弹性参数。

3. 对应的二维 nominal stress：
       sp(Lambda)/mu = (Lambda - Lambda^(-3)) * [
           1 + 2 c2 X + 3 c3 X^2 + 4 c4 X^3 + 5 c5 X^4
       ]

4. Transient-network 自由能：
       F(t) = exp(-beta t) Wp(lambda(t))
              + integral_0^t beta exp[-beta(t-t')]
                Wp(lambda(t)/lambda(t')) dt'

5. 指定拉伸历史：
       lambda(t) = exp(gamma_dot t)

   使用 tau = beta*t：
       q = gamma_dot / beta
       lambda(tau) = exp(q*tau)

6. 在该加载历史下：
       sigma/mu = lambda^(-a) sp(lambda)/mu
                    + a/lambda * integral_1^lambda
                      Lambda^(-a) sp(Lambda)/mu dLambda
       a = beta/gamma_dot = 1/q

注意
----
- 本模型不含 permanent-network fraction nu。
- 本模型没有 Gent 型有限伸长奇点。
- 高阶系数允许弹性曲线本身出现平滑的
  increase -> decrease -> increase 结构。
- 这是一个 phenomenological reduced-polynomial / extended-Yeoh
  弹性模型，不应预先赋予 c2~c5 过强的微观唯一含义。
- 默认参数仅用于展示和参数扫描，不代表实验拟合结果。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import warnings

from scipy.integrate import cumulative_trapezoid

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
save_path = "/home/tyt/project/Black_hole/5term_yeoh_transient_results"
os.makedirs(save_path, exist_ok=True)


# ====================================================================
#  五阶 Yeoh / reduced polynomial 弹性模型
# ====================================================================

def yeoh5_energy(lambda_val, c2, c3, c4, c5):
    """
    无量纲弹性自由能 Wp/mu。

    X = I1 - 2 = lambda^2 + lambda^(-2) - 2
    Wp/mu = 1/2 * [X + c2 X^2 + c3 X^3 + c4 X^4 + c5 X^5]
    """
    lam = np.asarray(lambda_val, dtype=np.float64)

    if np.any(lam < 1.0):
        raise ValueError("当前程序针对单轴拉伸，要求 lambda >= 1。")

    x = lam**2 + lam**(-2) - 2.0

    return 0.5 * (
        x
        + c2 * x**2
        + c3 * x**3
        + c4 * x**4
        + c5 * x**5
    )


def yeoh5_elastic_stress(lambda_val, c2, c3, c4, c5):
    """
    五阶 Yeoh 弹性 nominal stress / mu。

    sp/mu = (lambda-lambda^(-3)) *
            [1 + 2c2X + 3c3X^2 + 4c4X^3 + 5c5X^4]
    """
    lam = np.asarray(lambda_val, dtype=np.float64)

    if np.any(lam < 1.0):
        raise ValueError("当前程序针对单轴拉伸，要求 lambda >= 1。")

    x = lam**2 + lam**(-2) - 2.0

    bracket = (
        1.0
        + 2.0 * c2 * x
        + 3.0 * c3 * x**2
        + 4.0 * c4 * x**3
        + 5.0 * c5 * x**4
    )

    return (lam - lam**(-3)) * bracket


# ====================================================================
#  Transient-network 本构
# ====================================================================

def transient_stress(lambda_grid, q, c2, c3, c4, c5):
    """
    计算给定 q = gamma_dot/beta 下的 sigma/mu。

    使用：
        sigma = lambda^(-a) sp(lambda)
                + a/lambda * integral_1^lambda
                  Lambda^(-a) sp(Lambda) dLambda

    由于这里直接在 lambda-grid 上做一次 cumulative trapezoid，
    可用于任意弹性模型，不要求弹性应力具有解析积分形式。
    """
    lam = np.asarray(lambda_grid, dtype=np.float64)

    q = float(q)
    if q <= 0.0:
        raise ValueError("q = gamma_dot/beta 必须 > 0。")

    if np.any(lam < 1.0):
        raise ValueError("lambda 必须 >= 1。")

    a = 1.0 / q

    sp = yeoh5_elastic_stress(lam, c2, c3, c4, c5)

    # surviving chains
    survival = lam**(-a) * sp

    # re-crosslinked chains
    integrand = lam**(-a) * sp
    integral = cumulative_trapezoid(integrand, lam, initial=0.0)
    reborn = a * lam**(-1.0) * integral

    return survival + reborn


# ====================================================================
#  生成曲线
# ====================================================================

def compute_curve(q, lambda_max=3.0, n_points=1500,
                  c2=-0.75, c3=0.275,
                  c4=-0.0224, c5=2.68e-5):
    """
    输出 tau, lambda, sigma/mu。

    tau = beta*t
    lambda = exp(q*tau)
    """
    if lambda_max <= 1.0:
        raise ValueError("lambda_max 必须 > 1。")

    tau_max = np.log(lambda_max) / q
    tau = np.linspace(0.0, tau_max, n_points)
    lam = np.exp(q * tau)

    sigma_norm = transient_stress(lam, q, c2, c3, c4, c5)

    return tau, lam, sigma_norm


# ====================================================================
#  可视化：本构曲线
# ====================================================================

def plot_constitutive_curves(q_list, lambda_max,
                             c2, c3, c4, c5):
    fig, ax = plt.subplots(figsize=(12, 8))

    color_cycle = [
        '#d62728', '#1f77b4', '#2ca02c',
        '#ff7f0e', '#9467bd', '#8c564b'
    ]

    for i, q in enumerate(q_list):
        tau, lam, sigma_norm = compute_curve(
            q=q,
            lambda_max=lambda_max,
            n_points=1500,
            c2=c2,
            c3=c3,
            c4=c4,
            c5=c5,
        )

        ax.plot(
            lam,
            sigma_norm,
            '-',
            color=color_cycle[i % len(color_cycle)],
            linewidth=lines_linewidth,
            label=rf'$\dot{{\gamma}}/\beta={q:g}$'
        )

    ax.set_xlim(1.0, lambda_max)
    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sigma/\mu$', fontsize=label_fontsize)
    ax.set_title(
        '2D 5th-order Yeoh + Transient Network',
        fontsize=title_fontsize,
        pad=20,
    )

    ax.legend(fontsize=legend_fontsize * 0.9, loc='best')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(
        save_path,
        'yeoh5_transient_constitutive_curves.png'
    )
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'本构曲线已保存: {filepath}')


# ====================================================================
#  可视化：单独查看弹性本构
# ====================================================================

def plot_elastic_curve(lambda_max, c2, c3, c4, c5):
    lam = np.linspace(1.0, lambda_max, 1500)
    stress = yeoh5_elastic_stress(lam, c2, c3, c4, c5)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(lam, stress, '-', linewidth=lines_linewidth,
            label='Elastic Yeoh-5')

    ax.set_xlim(1.0, lambda_max)
    ax.set_xlabel(r'$\lambda$', fontsize=label_fontsize)
    ax.set_ylabel(r'$\sigma_p/\mu$', fontsize=label_fontsize)
    ax.set_title(
        '2D 5th-order Yeoh Elastic Response',
        fontsize=title_fontsize,
        pad=20,
    )
    ax.legend(fontsize=legend_fontsize * 0.9, loc='best')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(save_path, 'yeoh5_elastic_curve.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'弹性本构曲线已保存: {filepath}')


# ====================================================================
#  可视化：存活链 / 新生链 / 总应力
# ====================================================================

def plot_response_components(q, lambda_max,
                             c2, c3, c4, c5):
    tau, lam, total = compute_curve(
        q=q,
        lambda_max=lambda_max,
        n_points=1500,
        c2=c2,
        c3=c3,
        c4=c4,
        c5=c5,
    )

    a = 1.0 / q
    sp = yeoh5_elastic_stress(lam, c2, c3, c4, c5)

    survival = lam**(-a) * sp

    integrand = lam**(-a) * sp
    integral = cumulative_trapezoid(integrand, lam, initial=0.0)
    reborn = a * lam**(-1.0) * integral

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
        rf'Components at $\dot{{\gamma}}/\beta={q:g}$',
        fontsize=title_fontsize,
        pad=20,
    )
    ax.legend(fontsize=legend_fontsize * 0.9, loc='best')
    ax.grid(True, alpha=0.3)

    filepath = os.path.join(save_path, 'yeoh5_transient_components.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'响应分量图已保存: {filepath}')


# ====================================================================
#  主程序
# ====================================================================

def main():
    print('=' * 70)
    print('2D 5th-order Yeoh + Transient Network Constitutive Model')
    print('=' * 70)

    # ================= 【参数配置区】 =================

    # 动力学无量纲参数：q = gamma_dot / beta
    q_list = [0.1, 0.2, 0.4, 0.6, 0.8]

    # 变形范围
    lambda_max = 3.0

    # ------------------ 五阶 Yeoh 参数 ------------------
    # W/mu = 1/2 [X + c2 X^2 + c3 X^3 + c4 X^4 + c5 X^5]
    #
    # 当前默认值用于生成一个“软化-再硬化”的弹性形状，
    # 仅作为参数探索起点，不代表最终拟合。
    c2 = -0.75
    c3 = 0.275
    c4 = -0.0224
    c5 = 2.68e-5

    # ====================================================

    print('\n模型参数:')
    print(f'  c2 = {c2:.8e}')
    print(f'  c3 = {c3:.8e}')
    print(f'  c4 = {c4:.8e}')
    print(f'  c5 = {c5:.8e}')
    print(f'  lambda_max = {lambda_max:.6g}')
    print(f'  q=gamma_dot/beta = {q_list}')

    # 1. 弹性本构
    print('\n生成弹性本构...')
    plot_elastic_curve(
        lambda_max=lambda_max,
        c2=c2, c3=c3, c4=c4, c5=c5
    )

    # 2. 动态本构
    print('\n生成 transient-network 本构曲线...')
    plot_constitutive_curves(
        q_list=q_list,
        lambda_max=lambda_max,
        c2=c2, c3=c3, c4=c4, c5=c5
    )

    # 3. 分量分析
    print('\n生成响应分量图...')
    plot_response_components(
        q=0.5,
        lambda_max=lambda_max,
        c2=c2, c3=c3, c4=c4, c5=c5
    )

    # 4. 输出数据
    print('\n保存预测数据...')
    for q in q_list:
        tau, lam, sigma_norm = compute_curve(
            q=q,
            lambda_max=lambda_max,
            n_points=1500,
            c2=c2, c3=c3, c4=c4, c5=c5
        )

        data = np.column_stack((tau, lam, sigma_norm))
        filepath = os.path.join(
            save_path,
            f'yeoh5_q_{q:g}.txt'
        )

        np.savetxt(
            filepath,
            data,
            header=(
                'tau=beta*t, lambda=exp[(gamma_dot/beta)*tau], '
                'sigma_norm=sigma/mu\n'
                'tau lambda sigma_over_mu'
            )
        )

        print(f'  已保存: {filepath}')

    print('\n' + '=' * 70)
    print('计算完成！')
    print('=' * 70)


if __name__ == '__main__':
    main()
