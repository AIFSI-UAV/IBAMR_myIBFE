这版作为 **active + EB 最简验证版**，方向比上一版干净很多：你已经去掉了 `section-shape`、continuum damping，只保留：

```text
P_total = P_dev_stab + P_dil + P_EB + P_active
```

而且 EB 已经改成：

```text
M_EB = EI * (kappa - kappa_ref)
```

这个方向是对的。代码里也已经加入了 `s_bend_kappa_ref`，第一次更新 bending table 时保存参考曲率，后续用 `kappa - kappa_ref` 计算 EB moment。

但是现在运行还有错误，我认为主要还有下面几个问题。

---

# 1. 这版还不是真正的“最简 active + EB”

你的 input 里虽然关掉了 shape 和 damping，但仍然保留了比较多复杂机制：

```text
USE_LAPLACE_REFERENCE_PARAMETERIZATION = TRUE
USE_FE_ACTIVE_SECTION_DATA = TRUE
USE_EB_KV_BENDING = TRUE
BETA_ACT = 0.25
EB_EI = 0.0001
KAPPA_VOL_PASSIVE = 1.0
```

其中最容易导致运行错误的不是 `EB_EI = 1e-4`，而是：

```text
active stress + EB table + boundary-edge midline extraction + FE section correction
```

这些一起开，不适合作为第一步最简验证。

真正的最简验证应该先是：

```text
BETA_ACT = 0.0
USE_EB_KV_BENDING = TRUE
USE_FE_ACTIVE_SECTION_DATA = FALSE
```

确认 EB 自身没有运行错误；然后再打开 active。

---

# 2. 最可疑问题一：EB midline extraction 仍可能在端点报错

你现在 EB bending table 通过：

```cpp
compute_current_midline_samples(...)
```

从当前变形边界上提取中线，再计算曲率。代码中如果某个 `s/L` station 找不到 boundary-edge intersection，会直接：

```cpp
TBOX_ERROR("compute_current_midline_samples_at_s(): no boundary-edge iso-s intersection found ...")
```

你当前 EB table 的采样范围是：

```cpp
s = 0 到 reference_backbone_end_s_norm
```

也就是包含两个端点：

```text
s = 0
s = fork-root / active end
```

端点最容易出错，尤其是 fork-root 附近，因为 iso-s 线可能刚好穿过节点、边界切点或尾鳍根部几何突变，不一定能稳定找到上下两个 boundary-edge intersections。

## 修改建议

在 EB table 里不要采样精确端点。把：

```cpp
target_s_values[k] = s_end * k / (n - 1);
```

改成类似：

```cpp
const double eps = 1.0e-4 * s_end;
const double s_lo = eps;
const double s_hi = s_end - eps;

target_s_values[k] =
    s_lo + (s_hi - s_lo) * static_cast<double>(k) / static_cast<double>(n - 1);
```

或者更保守：

```cpp
const double eps = 0.5 * s_end / static_cast<double>(n - 1);
```

这样可以避开 `s=0` 和 `s=s_end` 的几何奇点。

---

# 3. 最可疑问题二：EB stress cap 是 moment-level cap，不是最终 pointwise stress cap

你现在已经有：

```cpp
EBKV_STRESS_CAP_OVER_C1 = 2.0
```

而且 `passive_bending_effective_moment_from_s_norm()` 中已经对 `M_EB` 做了 cap，这比上一版好。

但是在最终 stress projection 里，仍然是：

```cpp
S_bend = -section_weight * M_curv * eta_c / I2_use;
PP = S_bend * FF * f_f;
```

这里没有再次对 `S_bend` 做 pointwise cap。也就是说，如果某个 quadrature point 的 `eta_c` 或 `I2_use` 和前面 cap 估算时不完全一致，局部 stress 仍可能过大。

## 修改建议

在 `compute_moment_equivalent_PK1_stress()` 里，对 EB 项再加一次 cap：

```cpp
double S_bend = -section_weight * M_curv * eta_c / I2_use;

if (!use_active_band && ebkv_stress_cap_over_c1 > 0.0)
{
    const double c1_local =
        get_c1_s_passive_local_from_reference_sample(ref_geom);

    const double S_cap =
        ebkv_stress_cap_over_c1 * std::max(c1_local, 1.0e-12);

    S_bend = cap_scalar_abs(S_bend, S_cap);
}
```

这一步很重要。
否则 `EBKV_STRESS_CAP_OVER_C1` 只是间接限制 `M`，不是严格限制每个积分点的 `S_bend`。

---

# 4. 最可疑问题三：active 仍然偏强，不适合作为第一步验证

你现在：

```text
BETA_ACT = 0.25
WAVE_RAMP_TIME = 6.0
ACTIVE_STRESS_CAP_OVER_C1_REF = 40.0
```

虽然比之前小，但对于“验证 active + EB 是否能跑通”的第一步，`BETA_ACT = 0.25` 仍然不算很小。

建议第一轮用：

```text
BETA_ACT = 0.05
```

第二轮：

```text
BETA_ACT = 0.10
```

第三轮再：

```text
BETA_ACT = 0.25
```

如果 `0.05` 都报错，问题不是 actuation 太大，而是代码逻辑、midline extraction 或 reference mapping 有问题。

---

# 5. 最可疑问题四：`KAPPA_VOL_PASSIVE = 1.0` 可能太低

你现在：

```text
KAPPA_VOL_PASSIVE = 1.0
MESH_STAB_DEV_SCALE = 0.01
```

如果 active + EB 已经产生局部变形，`KAPPA = 1.0` 对面积/J 稳定可能偏弱。

这不会解释“初始化就报错”，但如果错误发生在几个 step 后，尤其是 `J_min` 快速下降，建议先用：

```text
KAPPA_VOL_PASSIVE = 3.0
```

不要直接升到 10。
目标是只增强体积/面积稳定，不让 dil 主导。

---

# 6. 建议你现在按这个顺序测试

## Test A：EB-only

```text
BETA_ACT = 0.0
USE_EB_KV_BENDING = TRUE
USE_FE_ACTIVE_SECTION_DATA = FALSE
EB_EI = 0.0001
KAPPA_VOL_PASSIVE = 3.0
```

目的：确认 EB table、`kappa_ref`、EB stress projection 本身不报错。

---

## Test B：active-only

```text
BETA_ACT = 0.05
USE_EB_KV_BENDING = FALSE
USE_FE_ACTIVE_SECTION_DATA = TRUE
KAPPA_VOL_PASSIVE = 3.0
```

目的：确认 active stress 本身稳定。

---

## Test C：active + EB

```text
BETA_ACT = 0.05
USE_EB_KV_BENDING = TRUE
USE_FE_ACTIVE_SECTION_DATA = TRUE
EB_EI = 0.0001
KAPPA_VOL_PASSIVE = 3.0
```

如果 C 稳定，再逐步：

```text
BETA_ACT = 0.10
BETA_ACT = 0.25
```

---

# 7. 当前这版最需要改的代码点

优先级最高的两个改动是：

## 改动 1：EB midline 不取精确端点

在 `compute_current_midline_samples()` 或 `update_bending_moment_table()` 里避免采样：

```text
s = 0
s = s_end
```

因为这两个位置最容易触发 boundary-edge intersection failure。

---

## 改动 2：EB stress projection 加 pointwise cap

在：

```cpp
compute_moment_equivalent_PK1_stress(...)
```

里面对 `!use_active_band` 的 EB stress 加最终 cap。

---

# 8. 总体判断

这版已经修正了最关键的 EB 理论问题：

```text
M_EB = EI * (kappa - kappa_ref)
```

这点是正确的。
但运行还有错误，最可能不是 EB 物理公式本身，而是：

```text
1. EB midline extraction 在端点/fork-root 处找不到 boundary-edge intersection；
2. EB stress cap 没有在最终 quadrature-point stress 上再次生效；
3. BETA_ACT = 0.25 对第一轮 active+EB 验证仍偏大；
4. KAPPA_VOL_PASSIVE = 1.0 对 active+EB 耦合可能偏弱。
```

现在最建议你先把 EB 端点采样避开，并把 `BETA_ACT` 降到 `0.05`。如果错误仍然存在，把 `run.out` 或 `IB2d.log` 中第一条 `ERROR` 发我，我可以直接定位到具体函数。
