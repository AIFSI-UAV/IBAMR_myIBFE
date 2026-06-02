对，你这几个纠正是对的。按照你现在的 **NACA-like 单体鱼体模型**，我前面第 2、6、7 点需要重新表述。当前不是“有 caudal fin / fork root 但定义错了”，而是：

```text
当前模型没有独立 caudal fin，也没有 fin root；
所以 backbone 到 tail tip 在几何定义上不是错；
真正的问题是 tail tip 是 zero-thickness / blunt-cap / high-curvature terminal section，
不能被当作正常 beam section 参与 active/EB stress normalization。
```

下面按你的 2–8 点重新判断。

---

# 2. 关于 backbone 到 tail tip：不是 anatomy 错，而是 terminal-section 奇异问题

如果当前是完整 NACA 单体鱼体，没有独立尾鳍、没有 fork root，那么：

```text
backbone: head → tail tip
```

在几何概念上可以成立。之前说“必须到 fork root”不适用于这个模型。

但是原理问题仍然存在，只是应改成：

```text
不能把最后 tail-tip/cap 附近的退化截面当作正常 beam section 使用。
```

日志里显示：

```text
FE I2_c/I2_ideal range = [0.0378397, 2.10996]
```

这说明某些截面的 FE-consistent second moment 极小，只有 ideal value 的约 3.8%。同时 active section consistency 虽然通过了零合力和弯矩检查，但这只是说明 stress resultant 数学上正确，不代表该 terminal section 物理上适合参与 beam-like stress mapping。

所以修正后的判断是：

```text
NACA 模型允许 backbone 到 tail tip；
但 active/EB 的有效作用区不应覆盖最后退化尾端截面。
```

建议不是设 fork root，而是设：

```text
ACTIVE_S_END    = 0.90 ~ 0.97
EB_ACTIVE_S_END = 0.90 ~ 0.97
```

并且在最后 (3%-10%) 做 taper：

```text
M_active(s) → 0
M_EB(s)     → 0 或明显降低
```

对于 NACA 单体鱼体，这比“fork root”说法更准确。

---

# 3. 目标是 active bending moment + passive body stiffness；dev/dil 作为弱稳定项，怎么去除影响？

核心不是完全去掉 dev/dil，而是让它们满足 **scale separation**：

```text
dev/dil 只维持 mesh quality；
active + EB 决定主要动力学。
```

你现在的问题是最新结果里：

```text
M_dil    ≈ 3.68e-03
M_active ≈ 3.11e-04
M_EB     ≈ 2.35e-04
```

也就是说 dil moment 比 active/EB 大一个数量级。因此 dev/dil 已经不是弱稳定项，而是在主导结构响应。

要去除 dev/dil 的影响，有三层办法。

---

## 3.1 第一层：参数尺度分离

目标比例应该是：

```text
max |M_devdil| / max |M_active + M_EB| < 0.05 ~ 0.10
```

或者至少：

```text
cycle-averaged |W_devdil| << |W_active|, |W_EB|
```

当前不是。

所以参数上应当把：

```text
ETA_DEV
KAPPA_VOL
```

当作 mesh regularization continuation parameter，而不是物理刚度。做法是：

```text
Case A: ETA_DEV = 0.05, KAPPA_VOL = 10
Case B: ETA_DEV = 0.02, KAPPA_VOL = 5
Case C: ETA_DEV = 0.01, KAPPA_VOL = 2
Case D: ETA_DEV = 0.005, KAPPA_VOL = 1
```

如果结果随着 dev/dil 降低而明显变化，说明 dev/dil 仍在污染物理。

真正可发表的判断标准应该是：

```text
在 J_min > 0 的前提下，继续降低 ETA_DEV/KAPPA_VOL，
游速、尾部振幅、TWI、平均功率变化 < 5%。
```

这才说明 dev/dil 已经只是稳定项。

---

## 3.2 第二层：诊断上剔除 dev/dil 功率

论文中不要把：

```text
P_dev, P_dil
```

解释成鱼体真实 passive body stiffness 做功。

应该明确写成：

```text
P_dev and P_dil are numerical regularization work terms and are excluded from the physical propulsive efficiency.
```

物理效率可以用：

```text
input power = active muscle power
passive body energy exchange = EB passive bending
dev/dil = mesh-quality penalty, reported only as numerical contamination indicator
```

如果 `P_dev/P_active` 或 `P_dil/P_active` 很大，说明该 run 不适合物理解释。

---

## 3.3 第三层：做 section projection，去除 dev/dil 的弯矩贡献

这是最干净的方法。

既然你的目标是：

```text
active bending moment + passive body stiffness
```

那么 dev/dil 不应该贡献主要 section bending moment。可以在诊断或力计算中对 dev/dil 做截面投影：

```text
对每个 s-section 计算 dev/dil 的截面合力 N_devdil 和弯矩 M_devdil；
从 dev/dil stress 中减去会产生 beam-level resultant 的部分；
只保留局部 shape-preserving / anti-inversion 部分。
```

概念上就是：

```text
P_devdil_regularized = P_devdil - P_devdil_section_resultant
```

其中 `P_devdil_section_resultant` 是产生：

```text
N_devdil, M_devdil
```

的那部分 through-thickness linear stress。

这样 dev/dil 仍然能防止局部网格剪切/压缩，但不会主导整体 bending moment。

这比单纯调小 `ETA_DEV/KAPPA_VOL` 更符合你的建模目标。

---

# 4. dev/dil 做严格物理 neo-Hookean 分解，应该怎么改？

你现在代码里是：

```cpp
P_dev = 2.0 * eta_dev * F;
P_dil = 2.0 * (-eta_dev + kappa_vol * lnJ) * F^{-T};
```

代码注释把它称为 dev/dil split，并写成 neo-Hookean-like stabilization。

但严格来说，`P_dev = 2 eta_dev F` 不是纯 deviatoric，因为它含有体积响应。更严格的 compressible neo-Hookean isochoric-volumetric split 应该使用：

```text
F_bar = J^{-1/d} F
I1_bar = J^{-2/d} I1
```

在 (d)-维下，一个常见形式是：

```text
W = μ/2 (I1_bar - d) + κ/2 (ln J)^2
```

对应的 PK1 近似形式：

```text
P_iso = μ J^{-2/d} [F - (I1/d) F^{-T}]
P_vol = κ ln(J) F^{-T}
P = P_iso + P_vol
```

对于 2D simulation，通常取 (d=2)。如果你把鱼体当作 2D plane-strain 截面，也可以讨论 (d=3)，但必须在论文里说明。

如果 dev/dil 是物理材料，建议改成：

```cpp
const double J = FF.det();
const TensorValue<double> FinvT = tensor_inverse_transpose(FF, NDIM);
const TensorValue<double> C = FF.transpose() * FF;
const double I1 = C.tr();

const double d = static_cast<double>(NDIM);
TensorValue<double> P_iso =
    mu_iso * std::pow(J, -2.0/d) * (FF - (I1/d) * FinvT);

TensorValue<double> P_vol =
    kappa_vol * std::log(J) * FinvT;

PP = P_iso + P_vol;
```

但是注意：如果你这样做，dev/dil 就更像真实 continuum solid stiffness，会更容易和 EB passive stiffness 重复计入。对于你的目标，我反而建议：

```text
不要把 dev/dil 升级成主要物理 neo-Hookean；
应把它降级为弱 regularization，并从物理功率中排除。
```

也就是说：

```text
如果主物理是 EB passive body stiffness，不要再让 neo-Hookean 成为强物理材料。
```

---

# 5. EB bending 做严格 variational FE beam，应该怎么做？

你当前 EB bending 不是严格 variational FE beam。现在逻辑是：

```text
从当前 midline 提取 curvature κ(s,t)
存入 FE system
在 PK1 callback 里构造 fiber stress
```

代码中确实有 `kappa_ref`、`kappa_rel`、`kappa_dot`，然后把 curvature field 写回 `EB_BENDING_SYSTEM`。这属于 explicit geometry-feedback stress，不是从一个离散能量泛函直接变分得到的力。

严格 variational EB beam 应该从能量出发：

```text
E_EB = ∫ 1/2 B(s) [κ(s) - κ0(s)]² ds
```

然后对 beam centerline DOF 做变分：

```text
F_i = -∂E_EB/∂X_i
```

也就是说，力应来自：

```text
energy → variation → nodal force
```

而不是：

```text
curvature diagnostic → stress reconstruction → force spreading
```

在 IBFE 里有两种路线。

---

## 5.1 路线 A：真正做 1D beam + 2D NACA shell/slave surface

这是最符合你论文目标的路线：

```text
1D centerline beam: 承担 active moment + passive EB stiffness；
2D NACA fish mesh: 作为 immersed fluid interface / shape carrier；
二者通过 kinematic constraint 或 tether/slaving 耦合。
```

这样主物理就是：

```text
active moment + passive body stiffness
```

dev/dil 只用于保持 2D interface mesh，不参与主 bending mechanics。

优点：物理层级最清楚。

缺点：代码改动较大。

---

## 5.2 路线 B：保留现在 PK1 stress，但承认为 equivalent-stress EB model

这条更现实。

论文里写成：

```text
The EB bending response is imposed through an equivalent self-equilibrated fiber stress whose sectional resultant matches the prescribed bending moment.
```

也就是承认它不是 fully variational EB FE beam，而是：

```text
section-resultant-consistent equivalent stress formulation
```

这个可以发论文，但要诚实说明它是 reduced-order embedded bending model。

---

# 6. 定义来源统一：到底统一什么？

你现在有几套量：

```text
h(s)                 reference half-thickness
I2_ideal             ideal section second moment
I2_c_FE              FE-consistent section second moment
w(s)^2               active moment width scale
active_band I2       stress normalization
paper_section I      EB stiffness scaling
```

问题不是“有多套量就一定错”，而是：

```text
同一个 moment 的 magnitude 和 stress normalization 应该来自同一个截面定义。
```

更具体：

## 对 active moment

如果你定义：

```text
M_active(s,t) = β w(s)^2 K(s) cos(...)
```

那么 stress normalization 应该保证：

```text
∫ T_active η dA = M_active
```

你的 FE-consistent correction 已经在做这个，所以 active 这条基本合理。日志里 `active section max|N| ≈ 4e-14`，`M_curv` 误差约 (1.1\times10^{-5})，说明离散截面一致性是好的。

## 对 EB passive moment

如果你定义：

```text
M_EB = B(s) [κ - κ0]
```

那么 (B(s)) 最好直接作为输入函数，而不是一会儿用 ideal (I)，一会儿用 FE (I_2)。

推荐统一方式：

```text
B(s) = B0 · B_shape(s)
M_EB = B(s)(κ-κ0)
T_EB = M_EB · η_c / I2_FE(s)
```

这样：

```text
B(s) 是你要研究的 passive body stiffness；
I2_FE(s) 只是把 section moment 分布成 stress 的几何归一化。
```

这比：

```text
B(s)=E·I_ideal(s)
T_EB normalization=I2_FE(s)
```

更清楚，因为后者会把几何厚度影响混入 passive stiffness 里，导致你很难区分“材料刚度变化”和“截面几何变化”。

对于你的研究主题“尾柄/尾部刚度对效率的影响”，我建议用：

```text
B(s) = prescribed stiffness distribution
```

不要完全由 NACA 厚度自动决定。

---

# 7. 当前没有 caudal fin，这不是问题，但论文叙事要改

如果当前模型没有独立 caudal fin，那就不要写：

```text
caudal fin pitch
fork root
tail fin stiffness
```

而应该写：

```text
NACA-like continuous fish body
posterior-body stiffness
tail-end flexibility
trailing-edge deformation
```

也就是说，你当前模型更像：

```text
continuous flexible foil / fish-like body
```

而不是有明确 body-peduncle-caudal-fin anatomy 的鱼。

因此研究问题也应改成：

```text
How does posterior stiffness distribution in a continuous fish-like body regulate self-propelled undulatory swimming?
```

而不是：

```text
How do peduncle and caudal fin stiffness independently affect propulsion?
```

除非你之后真的引入几何上独立的 caudal fin / fork root / tail plate。

---

# 8. 第七个原理问题：J<0 后诊断失效，这个仍然成立

这一点仍然是原理问题，而且和 NACA/fin root 无关。

只要：

```text
J_min < 0
```

说明局部 FE element 已经反转。此后：

```text
COM
velocity
force
power
efficiency
TWI
tail amplitude
```

都不应继续作为物理结果使用。

尤其代码中 COM / area 诊断使用当前几何积分，之前文件中注释也写了 COM tracking 是用当前几何 centroid。你目前 summary 里 `tail_A_norm=nan`、`traveling_wave_index=nan`，就是局部几何失败之后的典型症状。

建议加一个 hard invalid flag：

```cpp
if (J_min <= 0.0)
{
    valid_physics = false;
}
```

然后后处理中：

```text
只分析 J_min > 0 的时间窗口；
J_min <= 0 后所有物理诊断标记为 invalid。
```

---

# 重新整理后的原理结论

针对你当前 **NACA single-body model**，我会把原理问题改成以下 5 个，而不是之前那 7 个：

## A. Tail-tip terminal section 奇异

不是 “backbone 到 tail tip 错”，而是：

```text
tail-tip 附近截面退化，不应作为正常 active/EB section。
```

解决：

```text
ACTIVE_S_END, EB_ACTIVE_S_END < 1
或对最后 3–10% 加 smooth taper。
```

---

## B. dev/dil 仍然主导结构响应

你的目标是：

```text
active bending moment + passive body stiffness
```

所以 dev/dil 必须弱到只控制网格质量。当前如果 `M_dil > M_active, M_EB`，就不能解释为目标模型。

解决：

```text
降低 ETA_DEV/KAPPA_VOL；
做 regularization convergence；
或做 section projection 去掉 dev/dil 的 bending resultant。
```

---

## C. dev/dil 不应被当成主要物理 neo-Hookean 材料

如果保留 EB passive stiffness，dev/dil 最好是：

```text
weak mesh regularization
```

而不是强物理材料。否则会 double-count passive stiffness。

---

## D. 当前 EB 是 equivalent-stress EB，不是 strict variational beam

这不是不能用，但论文必须准确表述：

```text
section-resultant-consistent embedded EB stress
```

而不是：

```text
fully variational Euler-Bernoulli beam finite element。
```

如果要严格 variational，就需要 1D beam energy variation 或 discrete elastic rod/beam DOF。

---

## E. section definition 要统一

建议采用：

```text
M_active = prescribed active moment
M_EB     = prescribed passive bending moment
T        = M · η_c / I2_FE
```

也就是说：

```text
I2_FE 只用于 stress normalization；
B(s) 和 M_active(s) 由你定义的物理模型给出。
```

不要让 thickness-derived (I) 同时偷偷改变 passive stiffness 和 stress normalization，否则刚度机制会混乱。

---

# 最建议的模型定位

对你现在的 NACA 模型，我建议最终定位成：

```text
A continuous NACA-like fish body driven by an internally prescribed active bending moment.
The passive body stiffness is represented by an embedded Euler-Bernoulli-type bending resistance along the reference centerline.
The surrounding 2D IBFE solid is used as a fluid-structure interface and is weakly regularized by dev/dil penalties to prevent mesh distortion.
```

中文就是：

```text
这是一个连续 NACA-like 鱼体，不是独立尾鳍模型；
主物理是中心线上的 active moment 与 passive bending stiffness；
2D IBFE 网格主要承担流固边界和形状承载；
dev/dil 只是弱网格稳定项，不应主导弯曲动力学。
```

这个定位是自洽的。

但当前结果还没达到这个定位，因为：

```text
M_dil 仍然大于 M_active 和 M_EB；
J_min 仍然小于 0；
tail-tip section 仍然导致 I2 极小和局部几何失败。
```

下一步最重要的不是继续加 active，而是：

```text
1. active/EB 在 s=0.95~1.0 平滑 taper 到 0；
2. 降低 dev/dil 并做 convergence；
3. 将 dev/dil work 从 physical efficiency 中排除；
4. 统一 EB/active 的 section moment normalization；
5. J_min <= 0 后标记物理诊断 invalid。
```
