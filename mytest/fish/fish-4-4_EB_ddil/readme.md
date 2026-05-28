# fish4-4_EB.cpp：统一公式 → 显式 dev+dil 分离修改方案

**当前版本**：`ηtot(F − F⁻ᵀ)` 单参数（eta_tot = 0.05），体积约束不足，发散  
**目标版本**：显式 dev + dil 分离，`SPLIT_FORCES = TRUE`，EB 弯曲保持不变  
**发散根因**：EB 四阶 CFL + 体积约束不足，两者需同时修复

---

## 一、参数区修改（ModelData namespace）

### 1.1 删除

```cpp
// 删除
static double eta_tot = 0.05;
```

### 1.2 新增

```cpp
// ── 被动弹性：dev + dil 分离 ──────────────────────────────────────────
// dev：PP_dev = 2·eta_dev·FF
//   → 纯剪切抗力，防止单元剪切塌缩
//   → eta_dev 极小，不参与弯曲物理（bending 完全由 EB 负责）
//   → spurious/EB 比值 = 3·eta_dev/kappa_b ≈ 3×0.05/5 = 3%（可忽略）
static double eta_dev = 0.05;

// dil：PP_dil = 2·(−eta_dev + kappa_vol·lnJ)·F⁻ᵀ
//   → 独立体积约束，J→0 时发散阻止元素反转
//   → 与 dev 和 EB 完全解耦，不产生额外 spurious 力
//   → kappa_vol/eta_dev = 1000，充分近不可压
static double kappa_vol = 50.0;
```

### 1.3 force_decomp 诊断变量重命名

```cpp
// 修改前
static double s_force_decomp_work_passive      = 0.0;
static double s_force_decomp_work_weak_passive = 0.0;

// 修改后（区分 dev 和 dil）
static double s_force_decomp_work_dev          = 0.0;
static double s_force_decomp_work_dil          = 0.0;
static double s_force_decomp_work_weak_dev     = 0.0;
static double s_force_decomp_work_weak_dil     = 0.0;
```

### 1.4 保留不变

```cpp
// 以下全部保留，原样不动
static double c1_s_physics = 5.0;               // 主动应力上限基准
static double c1_s_physics_anterior = 5.0;
static double c1_s_physics_peduncle = 5.0;
static double c1_s_physics_caudal   = 5.0;
static double c1_s_body_transition_s  = 0.60;
static double c1_s_body_transition_w  = 0.30;
static double c1_s_caudal_transition_s = 0.85;
static double c1_s_caudal_transition_w = 0.10;
static double active_t_act_max_over_c1 = 200.0;    // 基准仍是 c1_s_physics
static double structural_kv_loss_factor = 0.05;    // 从 0.02 → 0.05（增强阻尼）
static double structural_kv_stress_cap_over_c1 = 50.0; // 基准仍是 c1_s_physics
static bool   use_eb_passive_bending = true;       // EB 保持开启
```

---

## 二、PK1 函数修改

### 2.1 删除统一被动函数

```cpp
// 删除整个函数
void PK1_passive_stress_function(TensorValue<double>& PP,
                                  const TensorValue<double>& FF, ...)
{
    PP = eta_tot * (FF - tensor_inverse_transpose(FF, NDIM));  // 删除
}
```

### 2.2 新增 dev 函数

```cpp
// ── Dev PK1：纯剪切抗力 ────────────────────────────────────────────────
// PP_dev = 2·eta_dev·F
// 对应应变能：W_dev = eta_dev·(I₁ - 3)
// 物理：抵抗形状变化（剪切），eta_dev 极小不干扰 EB 弯曲
// 注：当 J→0 时 PP_dev→0，不能单独防止体积塌缩（由 dil 负责）
void PK1_dev_stress_function(TensorValue<double>& PP,
                              const TensorValue<double>& FF,
                              const libMesh::Point& /*X*/,
                              const libMesh::Point& /*s*/,
                              Elem* const /*elem*/,
                              const vector<const vector<double>*>& /*var_data*/,
                              const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                              double /*time*/,
                              void* /*ctx*/)
{
    PP = 2.0 * eta_dev * FF;
    return;
}
```

### 2.3 新增 dil 函数

```cpp
// ── Dil PK1：独立体积约束 ──────────────────────────────────────────────
// PP_dil = 2·(−eta_dev + kappa_vol·lnJ)·F⁻ᵀ
// 对应应变能：W_dil = −2·eta_dev·lnJ + kappa_vol·(lnJ)²
// 物理：J→0 时 lnJ→−∞，F⁻ᵀ→∞，提供发散恢复力阻止元素反转
// 注：与 dev 合并即为标准 neo-Hookean（Holzapfel 2000, §6.4）
//       当 kappa_vol=0 时退化为水母论文的 ηtot(F−F⁻ᵀ)（beta=0 情形）
void PK1_dil_stress_function(TensorValue<double>& PP,
                              const TensorValue<double>& FF,
                              const libMesh::Point& /*X*/,
                              const libMesh::Point& /*s*/,
                              Elem* const /*elem*/,
                              const vector<const vector<double>*>& /*var_data*/,
                              const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                              double /*time*/,
                              void* /*ctx*/)
{
    const double J = FF.det();
    if (J <= 0.0)
    {
        // 元素已反转：返回大惩罚力推回
        PP = -2.0 * kappa_vol * tensor_inverse_transpose(FF, NDIM);
        return;
    }
    const double lnJ = std::log(J);
    PP = 2.0 * (-eta_dev + kappa_vol * lnJ)
             * tensor_inverse_transpose(FF, NDIM);
    return;
}
```

### 2.4 保留不变的函数

```cpp
// 以下函数完整保留，不修改
void PK1_active_stress_function(...)   { ... }   // 主动弯矩，不变
void PK1_eb_passive_bending_function(...){ ... }  // EB 弯曲，不变
void PK1_kv_damping_function(...)      { ... }   // KV 阻尼，不变
```

---

## 三、PK1 函数注册修改（main 函数中）

### 3.1 修改前（统一被动）

```cpp
ib_method_ops->registerPK1StressFcn(PK1_passive_stress_function, ...);
ib_method_ops->registerPK1StressFcn(PK1_active_stress_function,  ...);
ib_method_ops->registerPK1StressFcn(PK1_eb_passive_bending_function, ...);
ib_method_ops->registerPK1StressFcn(PK1_kv_damping_function,     ...);
```

### 3.2 修改后（dev + dil 分离）

```cpp
// dev → 体积力（volumetric body force）
ib_method_ops->registerPK1StressFcn(PK1_dev_stress_function,
    std::vector<unsigned int>(),
    std::vector<IBTK::SystemData>(),
    IBFE::IBFEMethod::PK1_STRESS_FCN_SYMMETRIC_DEV);   // DEV 标记

// dil → 界面压力跳变（surface force）
ib_method_ops->registerPK1StressFcn(PK1_dil_stress_function,
    std::vector<unsigned int>(),
    std::vector<IBTK::SystemData>(),
    IBFE::IBFEMethod::PK1_STRESS_FCN_SYMMETRIC_DIL);   // DIL 标记

// 以下不变
ib_method_ops->registerPK1StressFcn(PK1_active_stress_function,  ...);
ib_method_ops->registerPK1StressFcn(PK1_eb_passive_bending_function, ...);
ib_method_ops->registerPK1StressFcn(PK1_kv_damping_function,     ...);
```

---

## 四、input 文件修改

### 4.1 删除的参数

```
# 删除
ETA_TOT = 0.05
```

### 4.2 新增/修改的参数

```
# ── 被动弹性：dev + dil ──────────────────────────────
ETA_DEV   = 0.05        # dev 剪切刚度（极小，不干扰 EB）
KAPPA_VOL = 50.0        # dil 体积约束（原 20→50，薄体需要更强）

# ── IBFEMethod 配置 ───────────────────────────────────
IBFEMethod {
    SPLIT_FORCES = TRUE     # dev→体积力，dil→界面压力跳变
    # 其余参数保持不变
}

# ── 时间步：解决 EB 四阶 CFL ─────────────────────────
DT = 1.0e-4             # 从 5e-4 → 1e-4（EB 弯曲波稳定性要求）

# ── 结构阻尼：耗散 EB 振荡 ────────────────────────────
STRUCTURAL_KV_LOSS_FACTOR = 0.05   # 从 0.02 → 0.05

# ── 主动力斜坡：给足初始稳定时间 ────────────────────────
WAVE_RAMP_TIME = 5.0    # 从 3.0 → 5.0
```

### 4.3 不变的参数

```
# 以下完全保留
C1_S_PHYSICS        = 5.0      # 主动应力上限基准（不参与被动弹性）
BETA_ACT            = 5.0
ACTIVE_T_ACT_MAX_OVER_C1 = 200.0
USE_EB_PASSIVE_BENDING = TRUE
EB_KAPPA_B          = <现有值>
```

---

## 五、力学职责对比表

```
修改前（单参数）          修改后（dev+dil 分离）
─────────────────        ──────────────────────────
ηtot(F−F⁻ᵀ)             PP_dev = 2·eta_dev·FF
  ↑ 剪切 ← 0.05           ↑ 纯剪切，eta_dev=0.05
  ↑ 体积 ← 0.05（不足）   PP_dil = 2(-eta_dev+κvol·lnJ)·F⁻ᵀ
  ↑ 弯曲副效应（混入）      ↑ 独立体积约束，kappa_vol=50

EB 弯曲                  EB 弯曲（不变）
  ↑ 被动弯矩               ↑ 被动弯矩，kappa_b直接=EI

Active stress            Active stress（不变）
  ↑ 主动弯矩               ↑ 主动弯矩，beta_act驱动

KV 阻尼（loss=0.02）     KV 阻尼（loss=0.05，增强）
  ↑ 结构阻尼               ↑ 耗散EB振荡
```

---

## 六、发散原因与对应修复

| 发散原因 | 对应修复 |
|---|---|
| `eta_tot=0.05` 体积约束极弱，J→0 | `kappa_vol=50` 独立体积罚，J→0 时发散 |
| EB 四阶 CFL 要求小 DT | `DT=1e-4`（从 5e-4 降低 5×）|
| EB 振荡无阻尼放大 | `KV_LOSS=0.05`（增大 2.5×）|
| 初始瞬态力过大 | `WAVE_RAMP_TIME=5.0`（从 3.0 增大）|
| 薄体压力跳变不精确 | `SPLIT_FORCES=TRUE`，dil 以表面力传递 |

---

## 七、修改文件清单

```
fish4-4_EB.cpp → fish4-4_devdil.cpp（建议新文件名）
  参数区：删除 eta_tot，新增 eta_dev + kappa_vol
  函数区：删除 PK1_passive，新增 PK1_dev + PK1_dil
  注册区：修改注册方式，添加 DEV/DIL 标记
  force_decomp：重命名 work_passive → work_dev + work_dil

input2d_devdil（对应 input 文件）
  删除 ETA_TOT
  新增 ETA_DEV=0.05，KAPPA_VOL=50
  修改 SPLIT_FORCES=TRUE，DT=1e-4
  修改 KV_LOSS=0.05，RAMP_TIME=5.0
```

---

## 八、验证检查清单

运行后检查以下指标确认修复成功：

```
geometry_conservation_diag.csv：
  ✓ J_min > 0（无元素反转）
  ✓ area_rel_error < 5%

curvature_phase_diag.csv：
  ✓ max|kappa_body| < 20/L（物理合理范围）

force_decomposition_diag.csv：
  ✓ work_eb_passive < work_active（EB 是阻力，不是主导）
  ✓ dominant_component 不再是 eb_passive

direction_debug.csv：
  ✓ v_forward > 0 且量级合理（< 2 L/s）
```

---

*方案版本：v2.0 | 基于 fish4-4_EB.cpp → dev+dil 分离*  
*参考：Griffith & Luo (2017) §3.2；Holzapfel (2000) §6.4*