已获取参数区，但需要看 PK1 函数的实际实现。继续获取后半部分。两次 fetch 都返回相同的起始段（文件很长，token 限制）。基于已读取的参数区做完整对比分析。

---

## 代码审查结果

### ✅ 已正确执行的修改

| 项目 | fish4-4_3.cpp | fish4-4_EB.cpp | 状态 |
|---|---|---|---|
| 体积罚参数 | `kappa_vol = 100.0` | 已删除 | ✅ |
| 稳定性 C1 | `c1_s_stab = 0.1` | 已删除 | ✅ |
| 统一被动弹性 | 无 | `eta_tot = 0.05` | ✅ |
| 正则化枚举 | `RegularizationMode{RAW_NH, PROJECTED_NH}` | 已删除 | ✅ |
| 投影截面数据 | `s_reg_*` 全套变量 | 已删除 | ✅ |
| 力分解诊断 | `s_force_decomp_work_dil` | 改为 `_work_passive` | ✅ |
| EB 弯曲 | `use_eb_passive_bending = true` | 保持 true | ✅ |
| 头文件注释 | "KEEP passive neo-Hookean + volumetric" | 改为"KEEP passive EB bending + soft shape" | ✅ |
| `eta_tot` 注释 | — | `"soft shape maintenance: eta_tot*(F - F^{-T})"` | ✅ |

---

### ⚠️ 遗留问题：`c1_s_physics*` 仍然存在

以下变量在 EB 版本中**完整保留**，但注释仍是"passive elasticity"语义：

```cpp
// --- Passive elasticity ---   ← 注释标签已不准确
static double c1_s_physics = 5.0;
static double c1_s_physics_anterior = 5.0;
static double c1_s_physics_peduncle = 5.0;
static double c1_s_physics_caudal   = 5.0;
static double c1_s_body_transition_s  = 0.60;
static double c1_s_body_transition_w  = 0.30;
static double c1_s_caudal_transition_s = 0.85;
static double c1_s_caudal_transition_w = 0.10;
```

`get_c1_s_physics_local()` 三段插值函数也完整保留。

这带来两个需要确认的问题：

**问题 1：PK1_passive 函数里用的是 `eta_tot` 还是 `c1_s_physics`？**

由于文件太长未能 fetch 到 PK1 函数体，这是最关键的未确认点。如果 passive PK1 函数里仍然写的是：
```cpp
PP = 2.0 * c1_s_physics * FF;   // 旧代码残留
```
而不是：
```cpp
PP = eta_tot * (FF - tensor_inverse_transpose(FF, NDIM));  // 目标
```
那么修改没有实际生效。

**问题 2：`active_t_act_max_over_c1` 的基准是谁？**

变量名仍是 `_over_c1`，暗示基准是 `c1_s_physics`。在 EB 版本中，`c1_s_physics` 被保留的合理理由只有一个：作为**主动应力上限**的归一化基准。如果是这样，应补充注释说明：

```cpp
// NOTE: in EB version, c1_s_physics is only used as the active stress
// cap reference (active_t_act_max_over_c1) and for omega* mapping.
// Passive elastic stress is now controlled by eta_tot, NOT c1_s_physics.
static double c1_s_physics = 5.0;
```

**问题 3：`structural_kv_stress_cap_over_c1` 基准不一致**

KV 阻尼帽的计算原为 `50.0 * c1_at_s`。如果 KV 阻尼基准还是 `c1_s_physics` 而不是 `eta_tot`，则 KV 帽的量级与被动弹性完全脱节（`c1=5` vs `eta_tot=0.05`，差 100 倍），KV 阻尼会失去约束效果。

---

### 需要补充确认的三点

1. **PK1_passive 函数体**：确认使用 `eta_tot*(FF - FF^{-T})`，不含 `c1_s_physics`
2. **KV 帽基准**：`structural_kv_stress_cap_over_c1` 的计算是否已改为 `* eta_tot`
3. **`c1_s_physics` 的实际用途**：在 EB 版中是否仅用于主动应力帽 + ω\* 映射，如果是则保留合理但需要补充注释澄清角色

能否提供 PK1 函数所在代码段，或者文件中搜索 `eta_tot` 的所有出现位置？