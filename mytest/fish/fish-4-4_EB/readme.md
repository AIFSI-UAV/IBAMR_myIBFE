# fish4-4_3.cpp → 纯净版修改方案

**目标架构：** Active stress + EB 被动弯曲 + 单一 dev 函数（`ηtot(F − F⁻ᵀ)`）  
**依据：** 水母论文（Hoover et al. 2017）统一公式 + 官方 ex4 `beta_s=0` 惯例  
**原则：** 弯曲物理由 EB 负责；单一 dev 仅做形状维护；去除所有 dil/PROJECTED_NH 依赖

---

## 一、参数区修改（ModelData namespace）

### 1.1 删除的参数变量

```cpp
// ── 全部删除 ──
static double kappa_vol = 100.0;                     // dil 体积罚，不再需要
static double c1_s_stab = 0.1;                       // 独立稳定性 C1，合并到 eta_tot
static double c1_s_physics = 5.0;                    // 三段物理 C1，由 eta_tot 替代
static double c1_s_physics_anterior = 5.0;
static double c1_s_physics_peduncle = 5.0;
static double c1_s_physics_caudal   = 5.0;
static double c1_s_body_transition_s  = 0.60;
static double c1_s_body_transition_w  = 0.30;
static double c1_s_caudal_transition_s = 0.85;
static double c1_s_caudal_transition_w = 0.10;
static double omega_star_prescribed = ...;           // 依赖 E_m=4·C1_S_PHYSICS，删除
static bool   em_from_omega_star    = false;         // 同上

// RegularizationMode 枚举及相关变量全部删除
enum class RegularizationMode { RAW_NH, PROJECTED_NH };
static RegularizationMode regularization_mode = ...;
static int         s_reg_projection_bins = 0;
static std::string s_reg_projection_quad_order = "FIFTH";
static bool        s_reg_projection_ready = false;

// s_reg_* 截面投影数据结构全部删除
static bool        s_reg_section_data_built = false;
static std::vector<double> s_reg_eta_bar;
static std::vector<double> s_reg_I2_c;
static std::vector<double> s_reg_area_per_length;
static std::vector<double> s_reg_proj_dev_A;
static std::vector<double> s_reg_proj_dev_B;
static std::vector<double> s_reg_proj_dil_A;
static std::vector<double> s_reg_proj_dil_B;

// force_decomp 中的 dil 分量追踪变量删除
static double s_force_decomp_work_dil       = 0.0;
static double s_force_decomp_work_weak_dil  = 0.0;
```

### 1.2 新增参数变量

```cpp
// ── 替换为单一统一弹性模量 ──
// Pe = eta_tot * (FF - FF^{-T})
// 对应应变能: W = (eta_tot/2)[tr(F^T F) - 3 - 2 ln J]
// 小变形等效杨氏模量 E ≈ 4·eta_tot（参考 Holzapfel 2000）
static double eta_tot = 0.05;   // 形状维护，仅防剪切/厚度塌缩
                                 // 弯曲刚度完全由 EB (kappa_b) 提供
```

### 1.3 保留不变的参数

```cpp
// ── 以下参数原样保留 ──
static double structural_kv_loss_factor = 0.02;      // KV 阻尼
static double structural_kv_stress_cap_over_c1 = 50.0; // 阻尼帽（基准改为 eta_tot）
static bool   use_eb_passive_bending = true;          // EB 弯曲（必须保持 TRUE）
static double eb_active_s_end = 1.0;
// active_* 全部保留（主动弯矩参数不变）
// beta_act, active_wavelength_over_L, ... 等不变
// active_t_act_max_over_c1 保留（基准改为 eta_tot）
```

---

## 二、PK1 应力函数修改

### 2.1 删除的函数

```cpp
// 全部删除
void PK1_dil_stress_function(...) { ... }
void build_reg_section_data(...) { ... }
void apply_projected_nh_regularization(...) { ... }
```

### 2.2 修改 `PK1_dev_stress_function` → `PK1_passive_stress_function`

**修改前（fish4-4_3.cpp 的 dev 函数）：**
```cpp
void PK1_dev_stress_function(TensorValue<double>& PP,
                              const TensorValue<double>& FF, ...)
{
    const double c1 = get_c1_s_at_s(s_norm);    // 三段插值
    PP = 2.0 * c1 * FF;
    // ... PROJECTED_NH 分支 ...
}
```

**修改后（统一公式）：**
```cpp
// 重命名为 PK1_passive_stress_function，语义更清晰
void PK1_passive_stress_function(TensorValue<double>& PP,
                                  const TensorValue<double>& FF,
                                  const libMesh::Point& /*X*/,
                                  const libMesh::Point& /*s*/,
                                  Elem* const /*elem*/,
                                  const vector<const vector<double>*>& /*var_data*/,
                                  const vector<const vector<VectorValue<double>>*>& /*grad_var_data*/,
                                  double /*time*/,
                                  void* /*ctx*/)
{
    // Pe = eta_tot * (F - F^{-T})
    // 来源：W = (eta_tot/2)[I1 - 3 - 2 ln J]，参见 Holzapfel (2000) §6.4
    // 等效于水母论文 Hoover et al. (2017) Eq.(2.7)：Pe = ηtot(F - F^{-T})
    // 两项同时提供：剪切抗力（F 项）+ 软体积约束（-F^{-T} 项，J→0 时发散）
    // SPLIT_FORCES=FALSE 时此函数作为唯一被动 PK1 函数注册
    PP = eta_tot * (FF - tensor_inverse_transpose(FF, NDIM));
    return;
}
```

> **注意：** 选择 `ηtot(F − F⁻ᵀ)` 而非 `2·C1·FF` 的理由：
> - 前者包含内置软体积约束（`-F⁻ᵀ` 项），无需额外 `kappa_vol`
> - 与水母论文公式一致，有文献依据
> - 参数仅 `eta_tot` 一个，语义清晰（不涉及 `kappa_vol/C1` 比值选择）

### 2.3 保留不变的函数

```cpp
// 以下函数原样保留
void PK1_active_stress_function(...) { ... }   // 主动弯矩 PK1（Xu 2024）
void PK1_eb_passive_bending_function(...) { ... } // EB 被动弯曲 PK1
void PK1_kv_damping_function(...) { ... }      // KV 结构阻尼
```

---

## 三、PK1 函数注册修改（main 函数中）

### 3.1 修改前

```cpp
// 当前：分别注册 dev 和 dil（SPLIT_FORCES=TRUE）
ib_method_ops->registerPK1StressFcn(PK1_dev_stress_function,  ..., IBFE_DEV_FCN);
ib_method_ops->registerPK1StressFcn(PK1_dil_stress_function,  ..., IBFE_DIL_FCN);
ib_method_ops->registerPK1StressFcn(PK1_active_stress_function, ...);
ib_method_ops->registerPK1StressFcn(PK1_eb_passive_bending_function, ...);
ib_method_ops->registerPK1StressFcn(PK1_kv_damping_function, ...);
```

### 3.2 修改后

```cpp
// 纯净版：只注册统一被动 + 主动 + EB + 阻尼
// SPLIT_FORCES=FALSE → 所有 PK1 函数均以体积力形式 spread
ib_method_ops->registerPK1StressFcn(PK1_passive_stress_function, ...); // 统一被动
ib_method_ops->registerPK1StressFcn(PK1_active_stress_function,  ...); // 主动弯矩
ib_method_ops->registerPK1StressFcn(PK1_eb_passive_bending_function, ...); // EB 弯曲
ib_method_ops->registerPK1StressFcn(PK1_kv_damping_function, ...);     // KV 阻尼
// 删除：PK1_dil_stress_function 注册行
```

---

## 四、input 文件修改

### 4.1 删除的参数

```
# 删除
KAPPA_VOL           = 100.0
C1_S_STAB           = 0.1
C1_S_PHYSICS        = 5.0
C1_S_ANTERIOR       = 5.0
C1_S_PEDUNCLE       = 5.0
C1_S_CAUDAL         = 5.0
C1_S_BODY_TRANSITION_S = 0.60
C1_S_BODY_TRANSITION_W = 0.30
C1_S_CAUDAL_TRANSITION_S = 0.85
C1_S_CAUDAL_TRANSITION_W = 0.10
REGULARIZATION_MODE = "RAW_NH"
OMEGA_STAR          = ...
EM_FROM_OMEGA_STAR  = FALSE
```

### 4.2 新增/替换参数

```
# 替换
ETA_TOT = 0.05          # 统一被动弹性参数 ηtot（Pe = ηtot(F-F^{-T})）
                        # 仅做形状维护；弯曲由 EB_KAPPA_B 控制
```

### 4.3 IBFEMethod 数据库修改

```
IBFEMethod {
   # 删除
   SPLIT_FORCES = TRUE       →   SPLIT_FORCES = FALSE
   
   # 保留
   USE_CONSISTENT_MASS_MATRIX = TRUE
   IB_USE_NODAL_QUADRATURE = FALSE
   # ... 其他保持不变
}
```

---

## 五、辅助函数/代码块处理

| 函数/代码块 | 处置 |
|---|---|
| `get_c1_s_at_s(s_norm)` 三段插值函数 | **删除**（`eta_tot` 为常数，无需插值）|
| `build_reg_section_data(...)` | **删除** |
| `apply_projected_nh_regularization(...)` | **删除** |
| `RegularizationMode` 枚举 | **删除** |
| `omega_star` 反推 `C1_S_PHYSICS` 代码块 | **删除**（`eta_tot` 不参与 ω\* 映射）|
| `s_reg_*` 数据的 build/use 调用点 | **全部删除** |
| force_decomp 诊断中的 `dil` 列 | **删除**（输出 CSV 列头也对应修改）|
| EB 弯曲相关所有代码 | **完整保留** |
| 主动弯矩所有代码 | **完整保留** |
| KV 阻尼代码 | **保留**（`structural_kv_stress_cap_over_c1` 基准改为 `eta_tot`）|

---

## 六、KV 阻尼中的参数基准替换

KV 阻尼帽的表达式原为：

```cpp
const double t_cap = structural_kv_stress_cap_over_c1 * c1_at_s;
```

修改为：

```cpp
const double t_cap = structural_kv_stress_cap_over_c1 * eta_tot;
```

---

## 七、物理一致性说明

```
新模型的力学职责划分（完全解耦）：

  ┌─────────────────┬──────────────────────────────────────────┐
  │ 物理作用         │ 负责模块                                  │
  ├─────────────────┼──────────────────────────────────────────┤
  │ 被动弯曲刚度     │ EB 弯曲：M = κ_b · I₂ · (κ - κ₀)        │
  │                 │ 截面净力严格为零，无 spurious axial force  │
  ├─────────────────┼──────────────────────────────────────────┤
  │ 主动弯矩驱动     │ Active PK1：T_act·η/I₂·(f₀⊗f₀)·FF       │
  │                 │ 不变，参数不变                             │
  ├─────────────────┼──────────────────────────────────────────┤
  │ 形状维护         │ ηtot(F − F⁻ᵀ)：防剪切塌缩 + 软体积约束  │
  │ （剪切+软不可压）│ eta_tot 极小（0.05），不贡献弯曲物理      │
  ├─────────────────┼──────────────────────────────────────────┤
  │ 结构阻尼         │ KV 阻尼（不变）                           │
  └─────────────────┴──────────────────────────────────────────┘

与现有版本的等效条件：
  若 eta_tot << kappa_b·I₂/L²，则被动弯曲物理与原版 EB 贡献相比，
  eta_tot 的弯曲副效应可忽略不计。
  典型值：eta_tot=0.05，EB kappa_b=5.0，L=1，I₂~0.001 → EI_EB/EI_dev ~ 100:1。
```

---

## 八、修改文件清单

```
fish4-4_3.cpp → fish4-4_clean.cpp
  - 参数区：删除 ~15 个变量，新增 eta_tot
  - 函数区：删除 3 个函数，修改 1 个，保留其余
  - 注册区：删除 1 行（dil 注册）
  - 辅助调用：删除约 10 处 s_reg_* 调用点

input2d_clean（对应 input 文件）
  - 删除 ~12 个参数
  - 新增 ETA_TOT = 0.05
  - 修改 SPLIT_FORCES = FALSE
```

---

*方案版本：v1.0 | 基于 fish4-4_3.cpp 分析*