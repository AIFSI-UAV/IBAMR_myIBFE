你的判断很可能是对的，但要更精确地说：

**问题不一定是 EB 理论错了，而是当前版本的 EB 实现方式还没有形成一个能量一致、变分一致的 bending operator。**

当前代码虽然已经把模型拆成了：

[
P_{\mathrm{total}}
==================

P_{\mathrm{active}}
+
P_{\mathrm{phys,bending}}
+
P_{\mathrm{reg}}
]

并且注释里也明确写了：physical passive stiffness 来自 beam-level bending moment law，再转换为 zero-resultant axial PK1 stress，weak neo-Hookean stress 只作为 numerical mesh regularization。这个总体架构是对的。

但是你现在 Step 5/6 的现象说明：**这个 section-projected EB bending stress 还没有产生“正确的整体弯曲动力学”。**

---

## 1. 为什么说可能是 EB 实现问题？

你现在做的是：

[
M_p(s,t)
========

B(s)\left[\kappa(s,t)-\kappa_0(s)\right]
+
D(s)\dot{\kappa}(s,t)
]

然后转换成截面轴向应力：

[
T_{\mathrm{phys}}
=================

\frac{M_p}{I_2}\eta_c
]

再写成：

[
P_{\mathrm{phys,bending}}
=========================

T_{\mathrm{phys}}F(f_0\otimes f_0)
]

这个转换能保证局部截面意义上：

[
N_{\mathrm{phys}}\approx0
]

[
M_{\mathrm{phys}}\approx M_p
]

所以 Step 4 能通过。

但 EB beam 的真实力学不是只要求“截面弯矩对”，还要求沿中线的空间微分关系正确：

[
Q(s)=\frac{\partial M}{\partial s}
]

[
f_\perp(s)=\frac{\partial^2 M}{\partial s^2}
]

也就是说，**弯矩必须通过空间导数产生横向剪力和横向力密度。**

当前方法是把 (M_p) 转成 continuum axial stress，再依赖 IBFE 的 stress divergence 自动产生力。理论上可以近似这一点，但前提是：

1. (s,\eta) 场足够光滑；
2. (\kappa(s)) 计算足够平滑；
3. (M(s)) 沿中线连续；
4. 截面 (I_2(s)) 不出现局部突变；
5. stress divergence 与 beam-level (\partial_s^2 M) 等价。

你现在 Step 6 出现 `J_min=-52`，说明这些条件至少有一个没有满足。

---

## 2. 当前更像是“截面弯矩正确，但弯曲力不正确”

这句话很关键。

Step 4 通过说明：

[
M_{\mathrm{section}}
\approx M_{\mathrm{model}}
]

也就是局部截面 moment 对了。

但 Step 5/6 失败说明：

[
\nabla\cdot P_{\mathrm{phys,bending}}
]

产生的整体结构力不对，或者不稳定。

所以当前问题不是：

[
M_p=B\kappa+D\dot{\kappa}
]

这个文献公式不能用。

而是：

[
M_p
\rightarrow
T_{\mathrm{phys}}
\rightarrow
P_{\mathrm{phys,bending}}
\rightarrow
\nabla\cdot P
]

这条链条还没有证明等价于 EB beam bending force。

---

## 3. 为什么会“没有形成正确弯曲”？

我认为有四个主要原因。

### 原因 1：当前 EB 不是 variational EB

真正的 EB bending force 应该来自 bending energy：

[
E_b
===

\frac{1}{2}
\int B(s)\left[\kappa(s)-\kappa_0(s)\right]^2ds
]

被动力应该满足：

[
F_b
===

-\frac{\delta E_b}{\delta X}
]

这样能量上天然是回复的。

但你当前不是从能量泛函变分得到 force，而是：

1. 先用 finite difference 计算 (\kappa)；
2. 构造 (M_p)；
3. 再投影成 continuum stress；
4. 再由 stress divergence 产生 force。

这不是严格的能量一致离散化。

因此可能出现：

[
M_p \text{ 截面意义正确}
]

但：

[
F_b \text{ 不是 } -\delta E_b/\delta X
]

这会导致能量正反馈。

---

### 原因 2：曲率 finite difference 噪声进入 force，而不是只进入诊断

以前 (\kappa) 只是后处理，噪声只影响图。

现在：

[
\kappa
\rightarrow M_p
\rightarrow P_{\mathrm{phys,bending}}
]

所以曲率噪声直接变成结构力。

三点差分的放大率大约是：

[
1/\Delta s^2
]

你估计约 (3900\ \mathrm{m}^{-2})，这个量级足以把很小的 midline 噪声变成很大的 bending moment。

这说明：**当前 EB bending 不是在处理平滑 beam centerline，而是在处理 noisy midline samples。**

---

### 原因 3：(\dot{\kappa}) 项更危险

damping 项是：

[
D\dot{\kappa}
]

如果：

[
\dot{\kappa}
============

\frac{\kappa^n-\kappa^{n-1}}{\Delta t}
]

那么噪声又被：

[
1/\Delta t
]

放大。

所以 Step 6 中 damping 造成 `J_min=-52`，非常符合这个机制。

这不是简单的 damping 参数问题，而是说明当前 damping discretization 可能不是能量耗散型。

---

### 原因 4：截面 stress 投影不一定等价于 EB shear-force distribution

即使 (M_p) 对，转成：

[
T_{\mathrm{phys}}=\frac{M_p}{I_2}\eta_c
]

也只保证截面 moment 对。

但 EB bending 的横向力来自：

[
\partial_s^2 M
]

如果 (M(s))、(I_2(s))、(\eta_c(s))、(f_0(s)) 不光滑，stress divergence 会出现局部尖峰。特别是 fork-root、尾柄过渡区、低 (I_2) 区域，非常容易产生非物理局部力。

---

## 4. 怎么证明是不是 EB bending 没有形成正确弯曲？

现在不要直接看 swimming。应该做一个 **EB operator verification**。

### Test EB-1：modal sign test

给一个简单初始形状：

[
y(s,0)=A\sin(\pi s/L)
]

关闭 active，关闭 damping，只开 elastic bending：

[
M_p=B\kappa
]

然后计算：

[
K_{\mathrm{rms}}(t)
]

正确 EB elastic force 应该让该模态产生回复运动，而不是高频局部折叠。

如果出现局部尖峰增长，说明 EB operator 不对。

---

### Test EB-2：energy test

计算 bending energy：

[
E_b(t)
======

\frac{1}{2}
\int B(s)\left[\kappa-\kappa_0\right]^2ds
]

对于 elastic-only + fluid viscosity 情况，能量不应该持续增长。

对于 elastic + damping：

[
\frac{dE_b}{dt}
]

更应该下降。

如果 (E_b) 在 Step 6 中快速增长，说明当前 (P_{\mathrm{phys,bending}}) 不是能量稳定的 bending force。

---

### Test EB-3：beam force consistency test

离线比较两个量：

从 beam theory 得到：

[
f_\perp^{EB}(s)
\approx
\frac{\partial^2 M}{\partial s^2}
]

从 IBFE force decomposition 得到实际横向力：

[
f_\perp^{IBFE}(s)
]

如果两者空间分布完全不同，说明：

[
M_p \rightarrow P_{\mathrm{phys,bending}}
]

的 stress projection 没有形成正确 EB bending force。

---

### Test EB-4：mesh / station refinement sensitivity

改变：

```text
PASSIVE_BENDING_CACHE_STATIONS = 64, 128, 256
```

以及：

```text
smooth = 5, 10, 15
```

如果结果高度依赖 station 数和 smoothing 次数，说明当前 bending operator 主要受数值差分控制，而不是物理参数 (B(s)) 控制。

---

## 5. 如果确认是 EB operator 问题，应该怎么改？

有三条路线。按严谨程度排序。

---

### 路线 A：真正做 variational beam bending force

这是最严格的。

在 centerline 上定义 bending energy：

[
E_b
===

\frac{1}{2}
\int B(s)(\kappa-\kappa_0)^2ds
]

然后对 centerline nodes 做变分，得到 nodal bending force。

这类似 IBM 里传统 fiber/beam force 的做法。

优点：

* 能量一致；
* 回复方向自然正确；
* damping 可以单独用 velocity-proportional generalized force；
* 不依赖 noisy continuum stress divergence。

缺点：

* 需要额外实现 backbone force spreading；
* 和当前 continuum IBFE stress 框架耦合更复杂；
* 需要定义 centerline node 与体网格之间的力分配。

---

### 路线 B：保留 section stress，但用平滑的 spline/Savitzky–Golay 曲率

这是最实用的短期方案。

不要再用：

[
3\text{-point finite difference}
]

直接算 (\kappa)。

改成：

1. 对 midline (X(s)) 做局部多项式拟合；
2. 从拟合曲线计算：
   [
   \kappa=
   \frac{x_s y_{ss}-y_s x_{ss}}
   {(x_s^2+y_s^2)^{3/2}}
   ]
3. 对 (\dot{\kappa}) 使用低通滤波。

例如：

```text
window = 7 or 9
poly_order = 3
```

这比“3点差分 + 10次 smoothing”更像连续 EB 曲率。

---

### 路线 C：先去掉 (D\dot{\kappa})，只保留 elastic bending

如果当前目标是让 fish 能稳定形成 travelling wave，可以先设：

[
D(s)=0
]

只保留：

[
M_p=B(s)(\kappa-\kappa_0)
]

然后用流体黏性和 weak reg 提供耗散。

这是最稳妥的短期选择。文献中的 viscoelastic damping 可以后续再加，但不要让 (D\dot{\kappa}) 先把模型打崩。

---

## 6. 我建议你现在的判断应写成这样

你可以在记录里写：

> Step 5 confirms that the sign of the projected passive bending stress is correct. However, Step 6 reveals that the current EB implementation does not yet behave as an energy-stable bending operator. The issue is likely not the beam-level constitutive law (M_p=B\kappa+D\dot{\kappa}), but the discrete pathway from noisy finite-difference curvature to section-projected PK1 stress. The damping term amplifies curvature noise and causes mesh collapse.

中文就是：

**当前不是文献 EB 公式不适用，而是 EB 公式进入 IBFE 的离散实现还没有形成正确的弯曲算子。**

---

## 7. 下一步我建议这样做

优先级如下：

1. **先跑 elastic-only：**
   [
   D=0
   ]
   确认 (B\kappa) 是否能稳定产生回复弯曲。

2. **暂时不要使用 full (D\dot{\kappa})**，因为 Step 6 已经证明它会导致 catastrophic collapse。

3. **改曲率计算方法**：从 3 点差分改成 tangent-angle 或 local polynomial curvature。

4. **加 bending energy diagnostic**：
   [
   E_b=\frac12\int B(\kappa-\kappa_0)^2ds
   ]
   如果 (E_b) 在 passive-only 情况下增长，说明 EB operator 仍然不正确。

5. **只在 elastic-only EB 稳定后，再逐步恢复 damping。**

---

一句话总结：

**你的怀疑成立：现在最大的问题很可能不是 active，也不是 reg，而是当前 projected EB bending 没有形成能量一致的正确弯曲力。下一步不要急着调 damping；先验证并修正 EB bending operator 本身。**
