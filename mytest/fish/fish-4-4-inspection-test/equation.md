## 结论

你的检查报告大体正确，但有几处需要修正。基于当前 GitHub raw 版 `fish4-4_0.cpp`，我判断：

```text
架构方向：基本通过
EB backbone：已干净切断
P_matrix：公式正确
P_fiber：公式基本正确，但 ratio 含义不是 kf/μ，而是 kf/E
P_active：第一版可用，但仍是 -η/I2 线性截面应力，不是最终 muscle-band self-equilibrated stress
ref_geom 注册：已正确附加
force decomposition：参数保留了，但当前源码里没有明显完整实现 F_matrix/F_fiber/F_active 的分解输出
```

---

# 1. 你的报告中确认正确的部分

## 1.1 顶部注释和架构定位正确

当前文件顶部已经明确写成：

```cpp
// IBFE continuum fish with full physical material stresses.
// The passive body is a spatially varying continuum material calibrated from
// the target bending stiffness B(s). Active bending is represented by a
// self-equilibrated axial PK1 stress over each material section. No extracted
// centerline force or beam-specific time-step controller is used.
```

这个和当前源码实际架构一致：它不再是 strict EB backbone 版本，而是 full physical material 版本。([GitHub][1])

---

## 1.2 EB backbone 主路径确实已经移除

当前启动注册的是：

```cpp
PK1_matrix_stress_function
PK1_fiber_stress_function
PK1_active_stress_function
```

并且注册代码中使用了：

```cpp
ib_method_ops->registerPK1StressFunction(PK1_matrix_data);
ib_method_ops->registerPK1StressFunction(PK1_fiber_data);
ib_method_ops->registerPK1StressFunction(PK1_active_data);
```

没有看到 `registerLagBodyForceFunction(strict_eb...)` 或 physical bending PK1 被注册。这说明你这版已经切换成：

```text
P = P_matrix + P_fiber + P_active
```

而不是：

```text
EB/Kirchhoff backbone force + weak regularization
```

([GitHub][1])

---

## 1.3 `ref_geom` 系统已正确附加到三个 PK1 stress

你之前担心三个 PK1 函数是否正确附加 `REF_GEOM_SYSTEM_NAME`。这点已经可以确认：代码构造了

```cpp
const std::vector<SystemData> ref_geom_sys_data(
    1, SystemData(REF_GEOM_SYSTEM_NAME, ref_geom_vars));
```

然后分别传给：

```cpp
PK1_matrix_data
PK1_fiber_data
PK1_active_data
```

三个 PK1 callback 都通过 `reference_geometry_from_system_data(system_var_data)` 读取 `s, eta, t_hat`。因此这一点是通过的。([GitHub][1])

初始化顺序上，代码在 `initializeFEData()` 后调用 `fill_reference_geometry_system(mesh, equation_systems)`，再初始化 postprocessor 和 patch hierarchy；所以 `ref_geom` 数据在实际推进前会填充。([GitHub][1])

---

# 2. 你的报告中需要修正的部分

## 2.1 `fiber_stiffness_ratio` 不是 `kf/μ`，而是 `kf/E`

你的报告里说：

```text
如果 k_f = fiber_stiffness_ratio * μ(s)，那纤维刚度是基体剪切模量的 10%
```

但当前源码实际是：

```cpp
props.kf = fiber_stiffness_ratio * props.E;
```

启动日志也打印为：

```cpp
fiber stiffness ratio kf/E = ...
```

所以 `fiber_stiffness_ratio=0.10` 的含义是：

[
k_f = 0.1 E
]

不是：

[
k_f = 0.1\mu
]

因为：

[
\mu = \frac{E}{2(1+\nu)}
]

当 (\nu=0.45) 时：

[
\mu \approx \frac{E}{2.9}
]

所以：

[
k_f=0.1E \approx 0.29\mu
]

这仍然是合理的弱纤维增强，但解释必须改成：

```text
fiber_stiffness_ratio = kf / E
```

不是 `kf / μ`。([GitHub][1])

---

## 2.2 `material_nu_eff → K` 推导是正确的，但属于 3D effective bulk modulus

当前源码：

```cpp
props.mu = props.E / (2.0 * (1.0 + material_nu_eff));
props.K  = props.E / (3.0 * (1.0 - 2.0 * material_nu_eff));
```

所以这里使用的是常见 3D isotropic linear elasticity 关系：

[
\mu = \frac{E}{2(1+\nu)}
]

[
K = \frac{E}{3(1-2\nu)}
]

当 (\nu=0.45) 时：

[
\frac{K}{\mu}
=============

\frac{2(1+\nu)}{3(1-2\nu)}
\approx 9.67
]

所以你说“体积刚度约为剪切刚度 9 倍”基本正确。需要补一句：这是 **3D effective modulus relation**，而你的模型是 2D IBFE solid，所以论文中最好写成：

```text
effective area/volumetric modulus calibrated from an assumed ν_eff
```

而不是严格 3D bulk modulus。([GitHub][1])

---

## 2.3 `active_t_max_abs=50` 的判断基本对，但还要结合 (M_a\sim h^2)

当前 active stress 是：

[
T_a = -M_a\eta/I_2
]

并且代码用：

```cpp
T_active = active_t_max_abs * tanh(T_raw / active_t_max_abs)
```

做 soft cap。([GitHub][1])

你说 `active_t_max_abs=50` 很大，通常不起保护作用，这个判断基本正确。更精确地说：如果

[
M_a=\beta h^2 K(\xi)\cos(\cdots)
]

且

[
I_2=\frac{2h^3}{3}
]

在 (\eta \sim h) 处：

[
|T_a|
\sim
\frac{\beta h^2 h}{(2/3)h^3}K
=============================

1.5\beta K
]

而 `K_shape = 1 - cos(2πξ)` 的最大值是 2，所以：

[
|T_a|_{\max}\sim 3\beta
]

例如：

```text
BETA_ACT = 2  →  T_peak ~ 6
```

因此 cap=50 基本不会触发。作为调试保护，`5~10` 更合理；作为正式结果，最好验证 cap 不激活。

---

# 3. 当前代码中已经明确存在的主要问题

## 问题 1：`P_active` 仍是最简 (-\eta/I_2)，不是最终 self-equilibrated muscle-band

当前 active 函数核心是：

```cpp
const double I2_use = std::max(section_second_moment(h), section_second_moment_floor());
const double T_raw = -Ma * ref_geom.eta / I2_use;
PP = T_active * FF * f0_f0;
```

这说明主动应力仍然是：

[
T_a = -M_a \frac{\eta}{I_2}
]

这确实满足理想对称截面的：

[
\int T_a dA = 0
]

[
-\int T_a\eta dA = M_a
]

但这是在理想截面、理想 (\eta) 分布、理想 (I_2) 下成立。实际 FE 截面不一定对称，尤其头尾和尾柄处，(\eta) 分布可能偏移。因此当前所谓 self-equilibrated 是**解析近似意义上成立**，不一定在 FE 截面积分上严格成立。([GitHub][1])

建议下一步一定加 section diagnostic：

```text
N_active(s) = ∫ T_active dA
M_active_measured(s) = -∫ T_active η dA
M_active_target(s) = M_a(s,t)
```

判据：

```text
|N_active| / (|M_active|/h) < 1e-2
M_active_measured / M_active_target ≈ 1
```

没有这个诊断，现在不能确认 active stress 在真实 FE section 上是自平衡的。

---

## 问题 2：`section_i2_floor_ratio` 同时影响 passive material 和 active normalization

当前代码：

```cpp
props.I2 = max(section_second_moment(h), section_second_moment_floor());
props.E = props.B / props.I2;
```

active 中也使用：

```cpp
I2_use = max(section_second_moment(h), section_second_moment_floor());
```

所以 `section_i2_floor_ratio` 同时改变两件事：

```text
1. E = B/I2 的材料标定；
2. T_active = -Mη/I2 的主动应力强度。
```

这能防止尾部爆炸，但它也会改变目标 (B(s)) 和 (M_a(s,t)) 的严格映射。([GitHub][1])

这一点可以接受为第一版工程保护，但论文严谨版最好改成：

```text
1. active/material domain 不延伸到极薄尾尖；
或
2. 用 FE-section-normalized q(η) 代替 analytic I2；
或
3. 将 floor 的影响输出到 diagnostic，证明主要分析区域未触发 floor。
```

---

## 问题 3：`B(s) → E(s)` 只是 effective calibration，必须做静态弯曲校准

当前材料属性：

```cpp
E = B / I2
μ = E / [2(1+ν)]
K = E / [3(1-2ν)]
kf = ratio * E
```

这条链路是合理的。([GitHub][1])
但是实际 continuum body 的等效弯曲刚度还会受：

```text
mesh geometry
plane-strain/2D assumption
K/μ 比值
fiber stiffness
tail/head section irregularity
active/section floor
```

影响。

所以不能只因为写了 (E=B/I)，就声称 (B_{\rm eff}=B_{\rm target})。必须做一个 calibration test：

```text
BETA_ACT = 0
施加很小的 STATIC_MOMENT_M0 或外部小弯矩
测量 κ(s)
反推 B_eff(s)=M/κ
```

这个测试是 full material 版本能否和 fish stiffness 文献对应的关键。

---

# 4. 当前代码中可能被忽略的问题

## 4.1 force decomposition 变量保留了，但我没有看到完整分解实现

源码中保留了：

```cpp
s_force_decomp_diag_enable
s_force_decomp_diag_interval
s_force_decomp_diag_filename
```

启动日志也会打印 force decomposition diagnostics。([GitHub][1])

但是我在当前 raw 代码中没有找到类似：

```text
F_L1_matrix
F_L1_fiber
F_L1_active
P_matrix
P_fiber
P_active
```

的明确输出字段，也没有找到 `F_weak` / `P_abs` 等旧版 force decomposition 关键词。([GitHub][1])

这意味着两种可能：

```text
1. write_test_diagnostics 里只输出几何/中线/相位，没有真正分解 matrix/fiber/active；
2. force_decomp 名称保留了，但 full material 版的分量还没补齐。
```

这个需要你实际检查 `force_decomposition_diag.csv` 表头。如果没有 `matrix/fiber/active` 三类力和功率，建议优先补上。对 full material 版，最小应输出：

```text
F_L1_matrix
F_L1_fiber
F_L1_active
P_matrix
P_fiber
P_active
P_sum
```

否则你无法判断是 matrix、fiber 还是 active 在主导变形。

---

## 4.2 `P_fiber` 方向用了 `ref_geom.t_hat`，但没有显式重新归一化

`reference_geometry_from_system_data()` 中确实会把 `t_raw` 归一化得到 `geom.t_hat`，所以一般没问题。([GitHub][1])
但在 `compute_fiber_PK1_stress_impl()` 和 `compute_active_PK1_stress_impl()` 中，`f0` 直接取 `ref_geom.t_hat`。建议仍然保守加一次归一化 helper，避免某些插值误差或未来改动引入问题：

```cpp
VectorValue<double> f0 = ref_geom.t_hat;
const double nf = std::sqrt(f0*f0);
if (nf <= 1.0e-14) { PP = 0.0; return; }
f0 /= nf;
```

这不是当前致命 bug，但属于稳健性改进。

---

## 4.3 `active_s_smooth` 只处理纵向 envelope，不处理截面 muscle-band

当前 active 只是纵向平滑：

```text
active zone s/L = [ACTIVE_S_START, ACTIVE_S_END], taper = ACTIVE_S_SMOOTH
```

但截面方向仍是全截面线性分布 (-η/I2)。这和“肌肉带”物理并不完全一致。正式 full material 模型建议加入：

```text
ACTIVE_CROSS_SECTION_MODE = LINEAR_ETA / MUSCLE_BAND
ACTIVE_BAND_FRACTION = 0.3~0.5
```

其中 `MUSCLE_BAND` 用 FE section 数值归一化，确保零净力和目标弯矩。

---

## 4.4 `reference_centerline_end_x` 需要确认是否自动到尾尖

当前参数是：

```cpp
reference_centerline_end_x = NaN; // auto-detect fork root unless set
```

代码启动会打印 `reference centerline end x`。([GitHub][1])
如果自动检测结果是 `x=0` 尾尖，而你的几何存在 fork/root，full material 虽然不像 EB backbone 那样敏感，但材料标定和 active 区域仍会覆盖极薄尾端。这会影响：

```text
I2 floor 激活
E = B/I2
T_active = Mη/I2
```

建议明确设置 `REFERENCE_CENTERLINE_END_X = 0.10` 或根据实际几何确认自动检测值。

---

# 5. 对你原报告逐项修正

| 你报告中的点                                  | 我的判断                                                                                         |
| --------------------------------------- | -------------------------------------------------------------------------------------------- |
| 顶部注释正确                                  | 对                                                                                            |
| 参数命名重构清晰                                | 对                                                                                            |
| `ref_geom` 系统保留                         | 对                                                                                            |
| 几何初始化升级                                 | 对                                                                                            |
| 旧 EB backbone 移除                        | 对当前 GitHub raw 版基本成立                                                                         |
| `material_caudal_transition_s=0.85` 需确认 | 对，需确认是有意扩展 caudal 材料区                                                                        |
| `active_t_max_abs=50` 太大                | 对，建议先试 5~10                                                                                  |
| `fiber_stiffness_ratio` 需确认             | 已确认：是 `kf/E`，不是 `kf/μ`                                                                       |
| `material_nu_eff → K` 需确认               | 已确认：用 3D (K=E/[3(1-2ν)])，合理但需解释为 effective                                                   |
| `ref_geom` 是否附加到 PK1                    | 已确认：三个 PK1 都附加了                                                                              |
| PK1 函数是否存在                              | 已确认：matrix/fiber/active 都存在                                                                  |
| active 中 `time` 是否直接使用                  | 已确认：直接传入 `compute_active_PK1_stress_impl(..., time)`                                         |
| 还缺什么                                    | section-level active moment diagnostic、matrix/fiber/active force decomposition、muscle-band q |

---

# 6. 当前版本最终评价

这版已经可以作为 **full physical material 路线的第一版基准代码**。它已经完成了最重要的架构转变：

```text
旧路线：
EB backbone force + weak mesh regularization

新路线：
matrix material + fiber material + active material stress
```

但当前 `P_active` 仍然是解析 (-η/I2) 的第一版，不是最终的 FE-section-normalized muscle stress。当前最优先的下一步不是继续改 `P_matrix`，而是：

```text
1. 补 active section moment diagnostic；
2. 补 matrix/fiber/active force decomposition；
3. 把 active q 从 -η/I2 升级为 FE-section-normalized muscle-band q；
4. 做静态 bending calibration，验证 B_eff(s) 是否接近 B_target(s)。
```

如果这四步完成，这个版本才可以从“结构正确的第一版 full material model”升级为“论文级 full physical material fish model”。

[1]: https://raw.githubusercontent.com/AIFSI-UAV/IBAMR_myIBFE/refs/heads/master/mytest/fish/fish-4-4-inspection-test/fish4-4_0.cpp "raw.githubusercontent.com"
