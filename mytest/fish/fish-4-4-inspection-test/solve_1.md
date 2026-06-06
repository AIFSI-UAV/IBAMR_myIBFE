现在有了完整的材料。让我来系统地给你写出这个 full physical material 版本的完整设计方案。下面是完整的系统性回答，分为数学框架、IBFE实现规则、和与文献的对应三个层次。---

## 完整构建方案

### 一、数学框架

总 PK1 stress 的结构是：

```
P = P_matrix + P_fiber + P_active
```

每一项的物理来源和文献对应关系如下。

**P_matrix（被动软组织基体）**，对应水母文献（medusan mechanospace）中 mesoglea 的 neo-Hookean matrix：

```
W_matrix = (μ/2)(J^{-2/d} I₁ − d) + (K/2)(lnJ)²
P_matrix = 2μ J^{-2/d} (F − (I₁/d) F^{−T}) + K lnJ · F^{−T}
```

`μ(s)` 用来承载 bending stiffness。由 fish stiffness 文献中的 `B(s)` 反推：

```
E_target(s) = B_target(s) / I(s)
μ(s) = E_target(s) / (2(1+ν))    → 平面应变取 μ = E/4
I(s) = h(s)³/6   （2D 截面二阶矩）
```

这样 `B_eff(s) = E(s)·I(s)` 自然产生，无需 EB backbone。

**P_fiber（轴向纤维增强）**，对应鱼体沿身体方向的肌纤维/脊柱结构：

```
I₄ = f₀ · C f₀         （C = F^T F，f₀ 是 reference 轴向方向）
W_fiber = (k_f/2)⟨I₄−1⟩²   （⟨·⟩ = max(·,0)，只受拉）
P_fiber = 2 k_f ⟨I₄−1⟩ F(f₀⊗f₀)
```

`f₀` 直接取自已有的 `ref_geom.t_hat`，代码里已经按 Laplace 参数化存在每个 Gauss 点上。

**P_active（主动肌肉应力）**，这是核心创新，把 active bending 文献的 `Mₐ(s,t)` 转换成截面自平衡应力：

```
Mₐ(s,t) = β h(s)² K(s) cos(2πs/λ − ωt + φ₀)

构造截面形函数 q(η)，满足：
    ∫_A q dA = 0          （截面无净轴向力）
    −∫_A q·η dA = 1       （归一化弯矩）

最简版本：
    q(η) = −η / I₂,   I₂ = ∫η² dA ≈ 2h³/3

主动应力：
    T_a(s,η,t) = Mₐ(s,t) · q(η)

PK1 active：
    P_active = T_a · F(f₀⊗f₀)
```

---

### 二、IBFE 官方规则下的实现

IBFE 的 PK1 stress 注册接口签名（参考 ex3/ex4）：

```cpp
void PK1_stress_function(
    TensorValue<double>& PP,
    const TensorValue<double>& FF,
    const libMesh::Point& /*X_cur*/,
    const libMesh::Point& X_ref,
    Elem* const elem,
    const std::vector<const std::vector<double>*>& var_data,
    const std::vector<const std::vector<VectorValue<double>>*>& grad_var_data,
    double time,
    void* ctx
);
```

需要注册三个独立的 PK1 函数：

```cpp
// 1. P_matrix: decoupled iso+vol (已有骨架，升级 μ(s) 为位置相关)
IBFEMethod::PK1StressFcnData PK1_matrix_data(PK1_matrix_stress_function);
PK1_matrix_data.quad_order = FIFTH;
ib_method_ops->registerPK1StressFunction(PK1_matrix_data);

// 2. P_fiber: axial fiber reinforcement
IBFEMethod::PK1StressFcnData PK1_fiber_data(PK1_fiber_stress_function);
PK1_fiber_data.quad_order = FIFTH;
ib_method_ops->registerPK1StressFunction(PK1_fiber_data);

// 3. P_active: moment-equivalent active muscle stress
IBFEMethod::PK1StressFcnData PK1_active_data(PK1_active_stress_function);
PK1_active_data.quad_order = FIFTH;
ib_method_ops->registerPK1StressFunction(PK1_active_data);
```

关键实现：每个 Gauss 点需要知道 `(s, η, t_hat)`，这些已经存在 `ref_laplace_node_geom` 里（`REF_GEOM_S/ETA/T_X/T_Y` 系统）。`P_active` 函数读取当前 `time`，直接计算：

```cpp
void PK1_active_stress_function(
    TensorValue<double>& PP, const TensorValue<double>& FF,
    const libMesh::Point&, const libMesh::Point& X_ref,
    Elem* const, ..., double time, void*)
{
    const ReferenceGeometrySample ref_geom =
        interpolate_ref_geom_at_point(X_ref);   // 从已有系统插值

    const double s      = ref_geom.s;
    const double eta    = ref_geom.eta;
    const double h_s    = interpolate_halfthickness(s);
    const double I2     = (2.0/3.0) * h_s * h_s * h_s;   // 2D 截面 ∫η² dA

    // 纵向 active envelope（已有函数）
    const double env = longitudinal_active_envelope(s / ref_arc_length);
    if (env <= 0.0 || I2 < 1e-30) { PP = 0.0; return; }

    // Active moment Mₐ(s,t) — 与当前 EB 版本同形式
    const double Ma = compute_active_moment(s, time);   // = beta*h²*K(xi)*cos(phase)

    // 截面形函数 q(η) = -η/I₂
    const double q_eta = -eta / I2;

    // 主动轴向应力
    const double T_a = Ma * q_eta;

    // PK1: T_a * F(f₀⊗f₀)
    const VectorValue<double>& f0 = ref_geom.t_hat;
    const VectorValue<double> Ff0 = FF * f0;
    for (unsigned i = 0; i < NDIM; ++i)
        for (unsigned j = 0; j < NDIM; ++j)
            PP(i,j) = T_a * Ff0(i) * f0(j);
}
```

---

### 三、与两篇文献的对应关系

**水母文献（medusan mechanospace）**提供的思想：整个身体是主动/被动 continuum material，驱动力来自材料内部的主动收缩应力，不是外加 beam force。在鱼体上的对应：`P_matrix + P_fiber` = 被动 mesoglea-like body，`P_active` = 等效肌肉。

**fish stiffness 文献（body stiffness effects on propulsion）**提供的参数：`B(s)` 的分区分布（anterior/peduncle/caudal）直接用来反推 `μ(s)`，实现相同的刚度分布，只是力学路径从 EB 变为 continuum。

两者的连接量就是 `Mₐ(s,t)`：它在 EB 版本里进入 beam virtual-work residual，在新模型里进入 `T_a = Mₐ · q(η)` 的归一化，产生截面自平衡应力分布。形式上一样，力学路径完全不同。

---

### 四、需要注意的实现细节

`I₂` 在尾部很小，`q(η) = -η/I₂` 会放大。解决方法有两个：一是加 `I₂` 下限保护（`I₂ = max(I₂, epsilon)`），二是换成 muscle-band 形函数——在 `|η| > α·h(s)` 的区域才激活，数值归一化。后者更像真实肌肉分布，也更稳定。

`μ(s)` 的空间变化通过 `get_mu_local(X_ref)` 实现，与现有的 `get_c1_mesh_reg_local_from_reference_point` 同样的模式——用 `s` 坐标插值不同区域的材料参数。

`USE_STRICT_EB_VARIATIONAL_FORCE = FALSE` 和 `USE_PHYSICAL_BENDING = FALSE` 关闭 EB backbone，只保留三条 PK1 路径，这样两种模型可以在同一代码基础上通过参数切换对比。