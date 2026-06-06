## 总体评价

这版修改方向是对的，而且已经不是之前的 `strict EB backbone + weak regularization` 版本了。它基本实现了你说的 **full physical material fish**：

```text
P = P_matrix + P_fiber + P_active
```

并且源码开头已经明确写成：被动鱼体是由目标弯曲刚度 (B(s)) 标定的空间变材料；主动弯曲通过材料截面上的 self-equilibrated axial PK1 stress 表示；不再使用 extracted centerline force 或 beam-specific time-step controller。([GitHub][1])

也就是说，**主架构合格**。但是现在还不能说是最终版。最主要的问题是：`P_active` 仍然采用最简 (-\eta/I_2) 线性截面应力，没有真正升级到 muscle-band / FE-section-normalized active stress。因此它仍可能在尾部、小 (I_2) 区域重新出现应力放大。

---

# 1. 已经做对的地方

## A. EB backbone 已经完全移除

当前源码中没有 `USE_STRICT_EB_VARIATIONAL_FORCE` 和 `USE_PHYSICAL_BENDING`，启动注册时只注册了三类 PK1 stress：

```cpp
PK1_matrix_stress_function
PK1_fiber_stress_function
PK1_active_stress_function
```

并且启动日志写的是：

```text
IBFE continuum fish: full physical material
PK1 material = matrix + tension-only fiber + active
```

这说明你已经把物理主路径从：

```text
EB backbone force → 2D mesh
```

切换成：

```text
continuum material stress
```

这是对的。([GitHub][1])

---

## B. `P_matrix` 公式现在是正确的

当前代码：

```cpp
PP = props.mu * pow(J, -2.0 / d) *
     (FF - (I1 / d) * FinvT)
     + props.K * log(J) * FinvT;
```

这对应：

[
P_{\mathrm{matrix}}
===================

\mu J^{-2/d}
\left[
F-\frac{I_1}{d}F^{-T}
\right]
+
K\ln JF^{-T}
]

这个和你前面想要的 decoupled hyperelastic matrix 是一致的。([GitHub][1])

---

## C. (I(s)=2h^3/3) 已经改对

当前代码：

```cpp
inline double section_second_moment(const double halfthickness)
{
    const double h = std::max(halfthickness, 0.0);
    return (2.0 / 3.0) * h * h * h;
}
```

这说明你已经把 `h` 当作 half-thickness 处理，使用：

[
I_2(s)=\frac{2}{3}h^3
]

这是正确的。([GitHub][1])

---

## D. (B(s)\rightarrow E(s),\mu(s),K(s)) 的标定路径已经实现

当前：

```cpp
props.B = get_target_bending_B_local(s_norm);
props.I2 = max(section_second_moment(body_halfthick_from_s(s)),
               section_second_moment_floor());
props.E = props.B / props.I2;
props.mu = props.E / (2.0 * (1.0 + material_nu_eff));
props.K = props.E / (3.0 * (1.0 - 2.0 * material_nu_eff));
props.kf = fiber_stiffness_ratio * props.E;
```

这说明你的材料参数已经按：

[
E(s)=\frac{B(s)}{I_2(s)}
]

[
\mu(s)=\frac{E(s)}{2(1+\nu_{\mathrm{eff}})}
]

[
K(s)=\frac{E(s)}{3(1-2\nu_{\mathrm{eff}})}
]

设置。这个思路是合理的，属于 effective continuum calibration。([GitHub][1])

注意：这仍然是 **effective calibration**，不是严格材料实验标定。后面必须做静态弯曲测试验证 (B_{\mathrm{eff}})。

---

## E. `P_fiber` 做成了 tension-only fiber

当前 fiber：

```cpp
I4_minus_1 = Ff0 * Ff0 - 1.0;
if (I4_minus_1 <= 0.0) return;
PP = 2.0 * props.kf * I4_minus_1 * FF * f0_f0;
```

这对应：

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

这个是合理的 tension-only axial reinforcement。([GitHub][1])

---

# 2. 当前最主要的问题

## 问题 1：`P_active` 仍然是最简 (-\eta/I_2)，还不是 muscle-band active stress

当前 active：

```cpp
const double I2_use =
    std::max(section_second_moment(h), section_second_moment_floor());

const double T_raw = -Ma * ref_geom.eta / I2_use;

const double T_active =
    active_t_max_abs > 0.0
    ? active_t_max_abs * std::tanh(T_raw / active_t_max_abs)
    : T_raw;

PP = T_active * FF * f0_f0;
```

这说明 active stress 仍然是：

[
T_a = -M_a\frac{\eta}{I_2}
]

也就是你前面设计里的“最简版本”。([GitHub][1])

这个版本可以作为第一版验证，但它有两个风险：

```text
1. 尾部 h 小，I2 小，T_active 容易被放大；
2. 只有 analytic I2，不保证真实 FE section 上 ∫T dA=0 和 -∫TηdA=M 精确成立。
```

你加了：

```cpp
section_second_moment_floor()
active_t_max_abs * tanh(...)
```

这能防止数值爆炸，但它们都是工程保护。正式版本最好改成 **FE-section-normalized muscle-band q(η)**。

---

## 问题 2：`section_i2_floor_ratio` 会改变目标 (B(s)) 和 active moment normalization

当前：

```cpp
section_second_moment_floor()
```

用于 `props.I2` 和 active 的 `I2_use`。这样在尾部 (h) 很小的时候：

```text
真实 I2 被 floor 替代
→ E = B/I2 被降低
→ active stress T = Mη/I2 被降低
```

这能稳定尾部，但会改变目标材料刚度和主动弯矩映射。它可以保留作为 debug protection，但论文里必须说明，或者更好地改为：

```text
backbone/active domain 不进入极薄尾端
或
FE-section-normalized muscle-band q
```

否则尾部结果不是严格 (B(s)) calibration。

---

## 问题 3：`P_active` 的截面无净轴力只是近似成立

理论上：

[
\int_{-h}^{h}\left(-\frac{\eta}{I_2}\right)dA=0
]

但实际 FE 网格中：

```text
η 分布不一定完全对称；
Laplace 参数化的 eta 可能存在截面偏移；
截面不是理想 [-h,h] 直线；
尾部/头部截面很不规则。
```

因此实际可能出现非零净轴向 active force：

[
\int_A T_a dA \ne 0
]

这会导致 active 里混入 axial thrust / compression，而不是纯弯矩。

你需要新增一个 diagnostic：

```text
N_active_section = ∫ T_active dA
M_active_section = -∫ T_active η dA
M_target = M_a
```

检查：

```text
N_active_section ≈ 0
M_active_section / M_target ≈ 1
```

没有这个诊断，不能确认 active stress 真的是 self-equilibrated。

---

# 3. 需要优先补的功能

## A. 增加 active section diagnostic

建议输出：

```text
s_norm
A_section
eta_bar
I2_section
N_active
M_active_measured
M_active_target
M_ratio
```

判据：

```text
|N_active| / (|M_active|/h) < 1e-2
M_active_measured / M_active_target = 0.95 ~ 1.05
```

这个比看整体 force decomposition 更关键。

---

## B. 改成 FE-section-normalized q

不要长期用：

[
q=-\eta/I_2
]

建议做：

[
q(\eta)=a[g(\eta)-\bar g]
]

其中 (g(\eta)) 是 muscle-band shape，比如：

```text
g(η)=+1 上侧肌肉
g(η)=-1 下侧肌肉
g(η)=0 中间区
```

然后通过 FE section 积分归一化：

[
\bar g=\frac{\int g,dA}{\int dA}
]

[
a=-\frac{1}{\int [g-\bar g]\eta dA}
]

这样保证：

[
\int q,dA=0
]

[
-\int q\eta,dA=1
]

这一步是从“能跑的工程版”到“真正 full physical active material 版”的关键。

---

## C. 对 (B(s)\rightarrow E(s)) 做静态弯曲校准

当前只是按：

[
E(s)=B(s)/I(s)
]

设置材料。但实际 (B_{\mathrm{eff}}) 受以下因素影响：

```text
2D plane-strain/plane-stress 假设
K/μ 比值
fiber stiffness
mesh geometry
Laplace section definition
```

所以必须做一个 calibration case：

```text
BETA_ACT = 0
P_fiber = off 或固定
施加小静态 active moment M0
测量 κ(s)
反算 B_eff = M0 / κ
```

否则你不能断言 continuum material 已经复现目标 (B(s))。

---

# 4. 当前版本适合跑什么测试？

我建议先做下面三组，不要直接大 active 游泳。

## Test A：zero-force material stability

```text
BETA_ACT = 0
INITIAL_BEND_AMPLITUDE = 0
FIBER_STIFFNESS_RATIO = 0.0 或 0.1
END_TIME = 1.0
```

看：

```text
J_min
F_matrix
F_fiber
COM drift
```

期望：几乎不动。

---

## Test B：passive material relaxation

```text
BETA_ACT = 0
INITIAL_BEND_AMPLITUDE = 0.01
```

这次应该和 strict EB 冻结版不同：因为 material reference 是直线，当前弯曲形态应该产生 material restoring stress。看是否自然回弹。

---

## Test C：small active stress

```text
ACTIVE_MOMENT_MODE = TRAVELING
BETA_ACT = 0.05 ~ 0.1
WAVE_RAMP_TIME = 3.0
INITIAL_BEND_AMPLITUDE = 0
```

目标不是游很快，而是看：

```text
P_active 是否产生平滑 travelling curvature wave
J_min 是否 > 0.9
F_matrix/F_active 是否合理
```

---

# 5. 当前版本的结论

可以这样评价：

```text
架构：通过
P_matrix：基本正确
P_fiber：基本正确
B(s)→E(s)：实现了，但需要 calibration
P_active：第一版可用，但仍是 -η/I2 工程近似
EB backbone 移除：通过
诊断：还缺 section-level active moment diagnostic
```

所以我的判断是：

> **这版已经适合作为 full physical material 路线的第一版基准代码。**
> 但它还不是最终物理严谨版；下一步最重要的不是再改 matrix 或 fiber，而是验证并改进 `P_active` 的截面归一化，避免重新落回旧的 (\eta/I_2) 尾部放大问题。

[1]: https://raw.githubusercontent.com/AIFSI-UAV/IBAMR_myIBFE/refs/heads/master/mytest/fish/fish-4-4-inspection-test/fish4-4_0.cpp "raw.githubusercontent.com"
