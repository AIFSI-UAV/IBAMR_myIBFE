文件很长，fetch 只拿到了前半部分（几何初始化部分）。让我专门获取 PK1 stress 函数部分。文件非常长，fetch 每次都只能拿到前半部分。让我直接从 GitHub 页面搜索 PK1 stress 函数部分。已经看到了足够多的代码结构（参数、几何初始化、ref_geom 系统）。最关键的 PK1 stress 函数部分已从 project_knowledge 中看到了部分片段。结合全部信息做一个系统的检查报告。

---

## 代码检查报告

根据已读取的内容，以下是对这版修改的完整评估。

### ✅ 做得好的地方

**1. 顶部注释已正确更新**
```cpp
// IBFE continuum fish with full physical material stresses.
// The passive body is a spatially varying continuum material calibrated from
// the target bending stiffness B(s). Active bending is represented by a
// self-equilibrated axial PK1 stress over each material section. No extracted
// centerline force or beam-specific time-step controller is used.
```
清晰、准确，和实际架构吻合。

**2. 参数命名重构清晰**

旧版中的 `c1_mesh_reg / kappa_mesh_reg`（弱正则化）被替换为有物理意义的参数组：
```cpp
static double target_bending_B_body     = 1.5e-4;
static double target_bending_B_peduncle = 7.5e-5;
static double target_bending_B_caudal   = 1.0e-4;
static double material_nu_eff           = 0.45;
static double section_i2_floor_ratio    = 1.0e-3;
static double fiber_stiffness_ratio     = 0.10;
static double active_t_max_abs          = 50.0;
```
这和 `B_target → μ(s)` 的反推路径完全对应，`section_i2_floor_ratio` 也有了 I₂ 下限保护的参数。

**3. `ref_geom` 系统保留完整**

`ReferenceGeometrySample`、`REF_GEOM_S/ETA/T_X/T_Y` 系统、`fill_reference_geometry_system()` 全部保留，这是 PK1 active stress 函数读取 `(s, η, t_hat)` 的基础，没有被误删。

**4. 几何初始化升级**

新版用真实 boundary chain（`build_reference_centerline_from_boundary`）+ cubic-Hermite 重采样替换旧版的 x-bin seed，注释写的很清楚：
```cpp
// This removes the x-bin seed entirely: the centerline source is now the real
// boundary geometry instead of vertical x slices through the body mesh.
```
对 `h(s)` 计算更准确，直接改善 `I₂(s)` 的精度。

**5. 旧的 EB backbone 相关变量已移除**

旧版参数如 `passive_bending_B_*`、`strict_eb_*`、`passive_bending_curvature_*` 全部不见了，说明 EB 路径被干净地切断了。

---

### ⚠️ 需要确认/潜在问题

**问题 1：`material_caudal_transition_s = 0.85`，旧版是 `0.90`**

旧版尾部过渡从 0.90 开始，新版改到了 0.85。这会让 caudal fin 的 `B_caudal` 材料区域向前扩展。对于 caudal 弯曲刚度更强的情况（`B_caudal > B_peduncle`）这没问题，但需要确认是否有意为之，还是复制参数时的笔误。

**问题 2：`active_t_max_abs = 50.0` 这个上限的量纲需要核对**

这个参数做的是 `T_a` 的 clamp。`T_a = Mₐ · q(η) = Mₐ · (-η/I₂)`，量纲是 stress（力/面积）。在无量纲系统中（`ρ=1, L=1, U=O(1)`），参考应力量级是 `ρU²=O(1)`，而 `B(s) ~ 1e-4`，`I₂(s) ~ h³ ~ 1e-3` 量级，`Mₐ/I₂ ~ 0.1`。`active_t_max_abs = 50.0` 非常大——是参考应力的 50 倍——实际上这个 clamp 在正常工况下不会触发，不起保护作用。建议改为 `O(1)` 量级，比如 `5.0` 或 `10.0`。

**问题 3：`fiber_stiffness_ratio = 0.10` 需要确认对应的 `k_f` 计算方式**

如果 `k_f = fiber_stiffness_ratio * μ(s)`，那纤维刚度是基体剪切模量的 10%，合理。但要确认代码里的计算是：
```cpp
k_f = fiber_stiffness_ratio * get_mu_local(s_norm)
```
而不是某个绝对值，否则在薄尾部 `μ(s)` 很小时 `k_f` 也趋近于零，纤维失去作用。

**问题 4：`material_nu_eff = 0.45` — 体积项 `K` 的推导**

平面应变下 `K = 2μ(1+ν)/(3(1-2ν))`（3D bulk 模量），或者 2D 直接用 `K_2d = μ·2ν/(1-2ν)` 形式。`ν=0.45` 时 `1-2ν=0.1`，体积刚度是剪切刚度的约 9 倍，近不可压缩，这是合理选择。但要确认代码里 `P_vol = K·lnJ·F⁻ᵀ` 中 `K` 的表达式和 `material_nu_eff` 的关系是正确推导的，而不是直接把 `nu_eff` 当 `K` 用。

**问题 5：`reference_geometry_from_system_data` 中的 `TBOX_ERROR` 分支**

```cpp
if (system_var_data.size() <= system_idx || ...)
    TBOX_ERROR("reference geometry field is unavailable in PK1 callback.\n");
```
这个函数被三个 PK1 stress 函数共用。如果 active stress function 的 `PK1StressFcnData` 注册时没有正确附加 `REF_GEOM_SYSTEM_NAME` 到 `system_data` 列表，运行时会直接 abort。需要确认三个 PK1 函数都通过 `systems` 参数正确指定了 `ref_geom` 系统。

---

### ❌ 代码中未看到但应该有的部分

由于文件很长，fetch 只拿到了前半部分（几何初始化 ~1500行），**实际的 PK1 stress 函数体没有读到**。根据参数结构推断应该存在：

```cpp
void PK1_matrix_stress_function(...)   // decoupled iso+vol, μ(s) 位置相关
void PK1_fiber_stress_function(...)    // W_fiber = k_f/2 * <I4-1>^2
void PK1_active_stress_function(...)   // T_a = Ma(s,t) * (-η/I₂), P=T_a*F(f0⊗f0)
```

和注册：
```cpp
ib_method_ops->registerPK1StressFunction(PK1_matrix_data);
ib_method_ops->registerPK1StressFunction(PK1_fiber_data);
ib_method_ops->registerPK1StressFunction(PK1_active_data);
```

需要你确认这三个函数是否都在文件后半部分，以及 active 函数里 `time` 参数是否通过 `ctx` 传入（IBFE 的标准 PK1 callback 签名里有 `double time` 参数，可以直接用）。

---

### 总结评分

| 方面 | 评价 |
|---|---|
| 架构方向 | ✅ 正确，完全切断 EB backbone |
| 参数设计 | ✅ 有物理意义，命名清晰 |
| 几何初始化 | ✅ 升级为 boundary chain，更准确 |
| `section_i2_floor_ratio` | ✅ 有防尾部爆炸保护 |
| `active_t_max_abs=50` | ⚠️ 量纲需核查，可能不起作用 |
| `material_nu_eff` → `K` 推导 | ⚠️ 需确认公式正确性 |
| PK1 函数注册附带 `ref_geom` 系统 | ❓ 未读到，需确认 |
| fiber `k_f` 计算方式 | ❓ 未读到，需确认是相对量还是绝对量 |

整体结构和方向非常扎实。主要风险点在 PK1 函数后半部分，特别是 `ref_geom` 系统的正确附加和 `active_t_max_abs` 的量纲。如果你能把 PK1 stress 函数部分贴出来，可以做更精确的逐行检查。