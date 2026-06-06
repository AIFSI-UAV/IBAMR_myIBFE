## 总体判断

**这个框架是合适的，而且方向正确。**
它的核心思想可以成立：

```text
JFM 水母式 full physical material:
    P = passive continuum material + active material stress

active bending 文献驱动:
    prescribed travelling active bending moment M_a(s,t)

你的新模型:
    用 continuum active stress 实现 M_a(s,t) 的截面合弯矩
```

这比当前 `strict EB backbone + weak 2D regularization` 更像一个真实柔性体。当前代码本身也明确说明：active/passive bending 仍是 beam-level load，再通过 area-averaged backbone 转置分配回 2D mesh，而 continuum PK1 stresses 只是 weak mesh regularization。
你现在提出的新框架，就是要把这个逻辑换成 **continuum material 自己承载主动/被动应力**。方向对。

但是，你给出的方案里有几处需要修正。

---

# 1. `P_matrix` 公式需要修正一个系数

你写的是：

```text
W_matrix = (μ/2)(J^{-2/d} I₁ − d) + (K/2)(lnJ)²
P_matrix = 2μ J^{-2/d} (F − (I₁/d) F^{−T}) + K lnJ · F^{−T}
```

这里 **系数不一致**。

如果能量写成：

[
W_{\mathrm{iso}}
================

\frac{\mu}{2}
\left(
J^{-2/d}I_1-d
\right)
]

那么导数应是：

[
P_{\mathrm{iso}}
================

\mu J^{-2/d}
\left[
F-\frac{I_1}{d}F^{-T}
\right]
]

不是 (2\mu)。

也就是说，应改为：

```text
W_matrix = (μ/2)(J^{-2/d} I₁ − d) + (K/2)(lnJ)²

P_matrix =
    μ J^{-2/d} (F − (I₁/d) F^{-T})
    + K lnJ F^{-T}
```

如果你想保留代码里类似 `2*c1` 的形式，那么应写成：

[
W_{\mathrm{iso}}
================

c_1
\left(
J^{-2/d}I_1-d
\right)
]

[
P_{\mathrm{iso}}
================

2c_1J^{-2/d}
\left[
F-\frac{I_1}{d}F^{-T}
\right]
]

也就是：

```text
μ = 2c1
```

这点要统一，否则材料刚度会差一倍。

---

# 2. `I(s) = h³/6` 这里不对，取决于 h 的定义

你后面 active 里写：

```text
I₂ = ∫η² dA ≈ 2h³/3
```

这说明你的 (h) 是 **half-thickness**，即截面范围：

[
\eta\in[-h,h]
]

对于 2D per-unit-depth 截面：

[
I(s)
====

# \int_{-h}^{h}\eta^2,d\eta

\frac{2h^3}{3}
]

所以如果 `h(s)` 是 half-thickness，应使用：

```text
I(s) = 2 h(s)^3 / 3
```

不是：

```text
I(s) = h(s)^3 / 6
```

如果 (H) 是 full thickness，(H=2h)，则：

[
I = \frac{H^3}{12}
==================

# \frac{(2h)^3}{12}

\frac{2h^3}{3}
]

所以更安全的写法是：

```text
如果 h = half-thickness:
    I = 2 h^3 / 3

如果 H = full thickness:
    I = H^3 / 12
```

你当前代码里的 `ref_halfthickness` / `h_s` 明显更像 half-thickness，因此应使用：

```text
I = 2.0/3.0 * h_s^3
```

---

# 3. 由 `B(s)` 反推 `μ(s)` 的思路对，但要小心 plane strain / plane stress

你写：

```text
E_target(s) = B_target(s) / I(s)
μ(s) = E_target(s) / (2(1+ν)) → 平面应变取 μ = E/4
```

这里 “平面应变取 μ=E/4” 不严谨。

一般三维 isotropic elasticity 中：

[
\mu = \frac{E}{2(1+\nu)}
]

如果近似不可压缩：

[
\nu \approx 0.5
]

则：

[
\mu = \frac{E}{3}
]

不是 (E/4)。

在 2D IBFE 中你到底是 plane stress、plane strain，还是 purely 2D artificial solid，需要明确。对你的当前目标，建议先采用清晰的工程定义：

```text
E_target(s) = B_target(s) / I_2D(s)
μ(s) = E_target(s) / [2(1+ν_eff)]
ν_eff = 0.45 或 0.49
K(s) 由 ν_eff 或 K/μ 比值给定
```

例如：

[
K =
\frac{E}{3(1-2\nu)}
]

这是 3D 关系。如果你采用 2D area penalty，就不要强行说它是严格 3D bulk modulus，而应称为：

```text
effective area modulus
```

---

# 4. `P_active` 框架正确，但 `q(η)=-η/I₂` 只能作为第一版

你写的核心条件是对的：

[
\int_A q,dA=0
]

[
-\int_A q\eta,dA=1
]

然后：

[
T_a(s,\eta,t)=M_a(s,t)q(\eta)
]

[
P_{\mathrm{active}}
===================

T_a F(f_0\otimes f_0)
]

这个逻辑非常好。它保留了 active bending 文献的 (M_a(s,t))，但把它实现为 continuum 内部的 active stress。

但是，最简：

[
q(\eta)=-\frac{\eta}{I_2}
]

有两个问题。

---

## 4.1 尾部 (I_2) 小，仍会放大

你自己已经指出了：

```text
I₂ 在尾部很小，q = -η/I₂ 会放大。
```

这和旧的 `M η/I2` 问题是同源的。区别是现在 passive continuum 更强，可以部分承载，但尾端仍可能爆。

所以第一版可以用 analytic (I_2)，但正式版最好改成：

```text
按每个 s-station 的实际 FE section 数值积分得到 q(η) 的归一化
```

也就是不要只用：

[
I_2 = \frac{2h^3}{3}
]

而是用 FE section 上的积分：

[
A_0(s)=\int_A w_s,dA
]

[
A_1(s)=\int_A w_s\eta,dA
]

[
A_2(s)=\int_A w_s\eta^2,dA
]

然后构造零轴力、单位弯矩的 (q)。

---

## 4.2 更推荐的 (q)：去均值后的 muscle-band shape

可以定义一个原始 muscle-band shape (g(\eta))，例如只在上下侧激活：

```text
g(η) = +1  for η > αh
g(η) = -1  for η < -αh
g(η) = smooth transition otherwise
```

然后做截面归一化：

[
q(\eta)=a[g(\eta)-\bar g]
]

其中：

[
\bar g=\frac{\int_A g,dA}{\int_A dA}
]

保证：

[
\int_A q,dA=0
]

再令：

[
a =
-\frac{1}{\int_A [g(\eta)-\bar g]\eta,dA}
]

保证：

[
-\int_A q\eta,dA=1
]

这个比 (-\eta/I_2) 更像真实肌肉带，也更稳定，因为主动应力集中在肌肉区域，而不是整个截面线性分布。

---

# 5. `P_fiber` 方向正确，但要避免和 `P_active` 双重计算

你写：

[
W_{\mathrm{fiber}}
==================

\frac{k_f}{2}\langle I_4-1\rangle^2
]

[
P_{\mathrm{fiber}}
==================

2k_f\langle I_4-1\rangle F(f_0\otimes f_0)
]

这个是可以的，表示 passive axial fiber 只在拉伸时提供阻力。

但要注意：

```text
P_fiber 是被动结构支撑；
P_active 是主动肌肉应力；
二者都沿 f0 方向。
```

如果 (k_f) 太大，鱼会像轴向拉杆一样变硬，抑制 travelling curvature wave。建议第一版：

```text
k_f 不要太大
先让 matrix 承担主要实体支撑
fiber 只作为弱抗拉增强
```

经验上先让：

```text
F_fiber < 10%~30% F_active
```

不要让 fiber 主导动力学。

---

# 6. `P_matrix + P_fiber + P_active` 三个 PK1 函数注册是可行的

你写的 IBFE 实现方式基本可行：

```cpp
registerPK1StressFunction(PK1_matrix_data);
registerPK1StressFunction(PK1_fiber_data);
registerPK1StressFunction(PK1_active_data);
```

IBFE 会把多个 stress contribution 叠加进入结构力。这个结构清楚，也方便 force decomposition。

不过实现细节建议调整：

## 6.1 不要写 `interpolate_ref_geom_at_point(X_ref)`

你当前代码里已有 `REF_GEOM_SYSTEM_NAME` 和：

```cpp
reference_geometry_from_system_data(...)
```

PK1 callback 已经可以通过 `var_data` 读取：

```text
REF_GEOM_T_X
REF_GEOM_T_Y
REF_GEOM_ETA
REF_GEOM_S
```

所以在 PK1 stress function 里更应该直接用：

```cpp
const ReferenceGeometrySample ref_geom =
    reference_geometry_from_system_data(var_data);
```

而不是另外写一个 `interpolate_ref_geom_at_point(X_ref)`。

因为 `X_ref` 插值可能和 FE quadrature point 的系统数据不完全一致；而 `var_data` 是 IBFE 回调官方路径中传进来的 quadrature data。

---

## 6.2 `f0` 要归一化

你写：

```cpp
const VectorValue<double>& f0 = ref_geom.t_hat;
```

建议仍然保险归一化：

```cpp
VectorValue<double> f0 = ref_geom.t_hat;
const double nf = std::sqrt(f0*f0);
if (nf <= 1.0e-14) { PP.zero(); return; }
f0 /= nf;
```

避免 Laplace / interpolation 产生极小误差。

---

## 6.3 active stress 最好加 cap，但 cap 应作用在 (T_a)，不是 (M_a)

建议加入：

```text
ACTIVE_T_MAX
```

然后：

```cpp
T_a = clamp(T_a, -ACTIVE_T_MAX, ACTIVE_T_MAX);
```

或者 smoother：

```cpp
T_a = ACTIVE_T_MAX * tanh(T_a / ACTIVE_T_MAX);
```

但论文里要说明这是数值保护。正式结果最好验证 cap 不激活。

---

# 7. 与文献对应关系基本成立，但表述要更准确

你写的对应关系是对的，但建议这样表述：

## 水母文献对应

不是简单说：

```text
P_matrix + P_fiber = mesoglea-like body
```

更严谨是：

```text
The medusan model motivates the use of a continuum active-material formulation,
in which passive elasticity and active contraction are represented by
stress contributions within the material body.
```

因为水母没有 fish axial fiber / active bending moment，所以不能说一一对应。

## fish stiffness 文献对应

你说：

```text
B(s) 用来反推 μ(s)
```

这个逻辑成立，但要说成：

```text
B(s) is used to calibrate an effective continuum modulus distribution through
B_eff(s)=E(s)I(s).
```

也就是说，它不是直接等价，而是 **effective calibration**。

## active bending 文献对应

这个是最强的连接点：

```text
The same travelling active bending moment M_a(s,t) is retained, but it is
realized as a self-equilibrated active stress distribution over each body section.
```

这是你的创新核心。

---

# 8. 这个框架的最大风险

## 风险 A：2D continuum 自然弯曲刚度可能和目标 (B(s)) 不匹配

即使你设：

[
E(s)=B(s)/I(s)
]

真实数值中 (B_{\mathrm{eff}}) 还会受以下因素影响：

```text
1. mesh shape
2. 2D/plane strain assumption
3. incompressibility K
4. fiber stiffness
5. boundary/tail geometry
6. IB coupling and fluid loading
```

所以必须做一个 calibration test：

```text
static bending test:
    施加小端矩或小主动 moment
    测量 curvature
    反算 B_eff = M/kappa
```

只有这样才能确认 continuum material 的等效刚度真的接近文献 (B(s))。

---

## 风险 B：active stress 不一定自动产生 travelling curvature wave

在 EB 模型里，(M_a(s,t)) 直接进入 bending equation。

在 continuum 模型里，(M_a) 先变成 active stress，然后材料和流体共同决定曲率。因此实际 curvature wave 可能：

```text
相位滞后
幅值不足
局部变形
不沿尾部放大
```

所以需要诊断：

```text
M_a(s,t)
T_a(s,η,t)
actual curvature κ(s,t)
phase lag between M_a and κ
```

---

## 风险 C：active stress 正负号要重新验证

你定义：

[
-\int_A q\eta,dA=1
]

[
T_a=M_aq
]

这样得到：

[
-\int_A T_a\eta,dA=M_a
]

符号上是自洽的。
但实际鱼体正曲率方向还要和 `eta`、`t_hat`、`s` 方向一致。需要保留 `geometry_sign_diag` 或类似诊断。

---

# 9. 我建议的最终修正版框架

你的方案可以整理为下面这个更严谨版本：

```text
P = P_matrix + P_fiber + P_active

P_matrix:
    W_iso = μ(s)/2 * (J^{-2/d} I1 - d)
    W_vol = K(s)/2 * (lnJ)^2
    P_iso = μ(s) J^{-2/d} [F - I1/d F^{-T}]
    P_vol = K(s) lnJ F^{-T}

P_fiber:
    I4 = f0 · C f0
    W_fiber = kf(s)/2 * <I4 - 1>^2
    P_fiber = 2 kf(s)<I4 - 1> F(f0⊗f0)

P_active:
    M_a(s,t) = β h(s)^2 K_shape(s) cos(phase)
    construct q(s,η) numerically so that:
        ∫ q dA = 0
        -∫ q η dA = 1
    T_a = M_a q
    P_active = T_a F(f0⊗f0)
```

并且：

```text
USE_STRICT_EB_VARIATIONAL_FORCE = FALSE
USE_PHYSICAL_BENDING = FALSE
```

让 EB backbone 作为对照模型，不参与 full material 模型。

---

# 10. 最终评价

你的框架 **合适，而且有研究价值**。但要改 4 个关键点：

```text
1. 修正 P_matrix 的 μ 系数；
2. 修正 I(s)：若 h 是 half-thickness，用 I=2h^3/3；
3. active q(η) 第一版可用 -η/I2，但正式版建议改成 FE-section-normalized muscle-band q；
4. B(s) → μ(s) 只是 effective calibration，必须通过静态弯曲测试验证 B_eff。
```

一句话：
**这个框架可以作为你的“full physical material fish”主方案，但第一版不要追求太复杂；先实现 matrix + moment-equivalent active stress，关闭 EB backbone，做静态 bending calibration 和小幅 travelling active test。**
