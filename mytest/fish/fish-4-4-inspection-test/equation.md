对，这个方向更合理，也更有创新性：

> **用 JFM 水母文献的 full physical material 思想，让鱼体本身成为主动/被动 continuum material；同时保留 active bending 文献中的 travelling active bending moment 作为驱动目标。**

也就是说，你不是简单复现 Xu/Yu 那种 1D EB fish，也不是简单照搬水母的 active contraction，而是构造一种：

```text
continuum active-material fish
+ prescribed travelling active bending moment
+ emergent body curvature and swimming
```

这比现在的：

```text
strict EB backbone
+ weak 2D mesh regularization
```

更像一个真实柔性体。

当前源码里 active/passive bending 仍然是 beam-level virtual-work load，然后转置分配回 2D IBFE mesh；continuum PK1 stress 主要还是 weak regularization。
你现在的新想法，就是要把 2D continuum 从“承载 EB 力的薄片”升级为“主动/被动材料本体”。

---

# 1. 新模型的核心思想

新的力学路径应该从：

```text
active bending moment → EB backbone force → 2D mesh
```

改成：

```text
active bending moment → 等效肌肉主动应力分布 → continuum material deformation
```

也就是说，active bending moment 不再作为一条 beam force 直接拉鱼体，而是解释为：

> 上下/左右肌肉区域产生相反方向的主动轴向张力或压缩，从而在截面上形成一个合弯矩。

这样鱼体的变形来自整个材料体，而不是一条中线在拉一张薄膜。

---

# 2. 数学上怎么写？

总 PK1 stress 可以写成：

[
P =
P_{\mathrm{matrix}}
+
P_{\mathrm{fiber}}
+
P_{\mathrm{active}}
]

其中：

```text
P_matrix  = 被动软组织基体
P_fiber   = 被动纤维/轴向增强
P_active  = 主动肌肉应力
```

---

## 2.1 被动 matrix

可以继续用 decoupled hyperelastic material：

[
W_{\mathrm{matrix}}
===================

\frac{\mu}{2}(\bar I_1-d)
+
\frac{K}{2}(\ln J)^2
]

对应：

[
P_{\mathrm{matrix}}
===================

P_{\mathrm{iso}}
+
P_{\mathrm{vol}}
]

这部分让鱼体有基本的：

```text
抗剪
抗压缩
抗局部面积变化
```

---

## 2.2 被动 fiber / axial support

鱼体不是普通橡胶块，沿身体方向有明显轴向结构。可以加入弱 fiber reinforcement：

[
I_4 = f_0 \cdot C f_0
]

[
W_{\mathrm{fiber}}
==================

\frac{k_f}{2}\langle I_4-1\rangle^2
]

[
P_{\mathrm{fiber}}
==================

2\frac{\partial W_{\mathrm{fiber}}}{\partial I_4}
F(f_0\otimes f_0)
]

其中 (f_0) 是 reference body-axis direction。

这一步会让鱼体更像“有肌纤维/脊柱方向支撑的柔性体”，而不是一张纸。

---

## 2.3 主动 bending moment 变成主动肌肉应力

文献中的 active bending moment 形式一般是：

[
M_a(s,t)
========

M_0 K(s)\cos(ks-\omega t+\phi_0)
]

你不需要把它直接作为 EB moment force，而是构造一个截面主动应力 (T_a(s,\eta,t))，让它满足：

[
\int_A T_a , dA = 0
]

[
-\int_A T_a \eta , dA = M_a(s,t)
]

然后：

[
P_{\mathrm{active}}
===================

T_a(s,\eta,t)F(f_0\otimes f_0)
]

这样主动应力在截面上没有净轴向力，但有目标弯矩。

这和你以前的 (T=M\eta/I_2) 有相似性，但区别很关键：

```text
以前：
    2D body 很弱，active/EB stress 像外力一样拉扯网格。

新模型：
    active stress 是材料内部肌肉应力；
    passive material 同时承载和分散这个应力；
    不再需要强 EB backbone 把力集中传回 2D mesh。
```

---

# 3. 这个模型和当前 strict EB 版本的区别

| 项目         | 当前 strict EB 版本          | 新 full physical material + active bending         |
| ---------- | ------------------------ | ------------------------------------------------- |
| 主要弯曲来源     | 1D EB/Kirchhoff backbone | continuum active/passive material                 |
| 主动输入       | beam-level active moment | continuum active muscle stress，截面合成 active moment |
| 被动弯曲       | (B(s)(\kappa-\kappa_0))  | matrix + fiber + geometry 自然产生                    |
| 2D body 作用 | weak regularization      | 真实/弱真实材料本体                                        |
| 失稳风险       | 中线力拉裂薄片                  | 力分布更均匀                                            |
| 与 EB 文献对应  | 直接                       | 通过截面合弯矩 (M_a) 对应                                  |
| 创新性        | 中                        | 更高                                                |

---

# 4. 你可以保留 active bending 文献的驱动方式

重点是：**不要丢掉 (M_a(s,t))**。

你仍然可以使用：

[
M_a(s,t)
========

\beta h(s)^2 K(s)
\cos\left(
2\pi \frac{s}{\lambda}
----------------------

\omega t
+
\phi_0
\right)
]

但是它不再进入：

```text
EB residual
```

而是进入：

```text
active stress normalization
```

也就是：

[
T_a(s,\eta,t)
=============

M_a(s,t) q(\eta)
]

其中 (q(\eta)) 满足：

[
\int_A q(\eta)dA=0
]

[
-\int_A q(\eta)\eta dA=1
]

这一步把 “active bending moment” 变成了“肌肉应力分布”。

---

# 5. 具体实现路线

## Phase 1：保留当前 strict EB 版本作为对照

不要删掉当前版本。它仍然是很好的对照组：

```text
Model A:
strict EB/Kirchhoff backbone + weak continuum support
```

然后新建：

```text
Model B:
full physical material + active bending stress
```

两者对比会很有论文价值。

---

## Phase 2：关闭 passive EB backbone，只保留 active continuum stress

先做：

```text
USE_STRICT_EB_VARIATIONAL_FORCE = FALSE
USE_PHYSICAL_BENDING = FALSE
```

但不要回到旧的弱 `active+dev/dil`。而是使用：

```text
passive continuum material
+ active moment-equivalent muscle stress
```

即：

[
P =
P_{\mathrm{matrix}}
+
P_{\mathrm{fiber}}
+
P_{\mathrm{active}}
]

---

## Phase 3：构造 active stress shape

用截面坐标 (\eta) 定义上下/左右肌肉分布。

例如最简单版本：

[
q(\eta)
=======

-\frac{\eta}{I_2}
]

但这个容易在薄尾部放大。更物理的版本是 muscle-band distribution：

```text
dorsal side: +T
ventral side: -T
middle region: 0 或平滑过渡
```

然后数值归一化，使：

[
-\int_A q(\eta)\eta dA=1
]

这样可以避免简单 (\eta/I_2) 在薄区爆炸。

---

## Phase 4：被动材料提供弯曲刚度

不要再用强 EB passive bending。被动弯曲刚度应由 continuum material 自然产生：

[
B_{\mathrm{eff}}(s)
\sim
E(s)I(s)
]

如果你想保留文献中的 (B(s))，可以反推材料模量：

[
E(s)
====

\frac{B_{\mathrm{target}}(s)}{I(s)}
]

然后让不同区域的 continuum material 具有不同 (E(s))、(\mu(s))、(K(s))。

这样就能把 EB 文献中的 stiffness distribution 转换成 continuum material distribution。

---

# 6. 这就是创新点

你的新模型可以这样概括：

```text
Instead of prescribing the fish kinematics or applying a beam force directly,
we embed the active bending moment into a continuum active material model.
The prescribed travelling bending moment is realized as a self-equilibrated
muscle stress distribution across each body section, while the passive
deformation resistance is supplied by a hyperelastic/fiber-reinforced fish body.
```

中文就是：

> 不是直接给鱼体规定运动，也不是用一条 EB 梁强行拉动 2D 网格；而是把文献中的 travelling active bending moment 转化为鱼体材料内部的自平衡主动肌肉应力分布。鱼体的被动抵抗由 continuum hyperelastic / fiber material 提供。

这个方向比“简单复现文献”更强。

---

# 7. 最推荐的最终模型结构

## Model 2：active-material fish model

```text
continuum hyperelastic fish body
+ fiber reinforcement
+ moment-equivalent active muscle stress
```

目的：

```text
提出你自己的 full physical material fish swimming model
```

两者之间通过下面量连接：

[
M_a(s,t)
]

[
B_{\mathrm{target}}(s)
]

[
B_{\mathrm{eff}}(s)=E(s)I(s)
]

这样你的论文逻辑会非常清楚：

```text
先用 EB benchmark 验证 active bending moment 机制；
再提出 continuum active-material extension；
比较二者在 deformation, wake, efficiency, J stability 上的差异。
```

---

# 8. 一句话总结

你的想法是正确的：

> **用 JFM 水母文献的 full physical material 思想承载鱼体变形，用 active bending 文献的 travelling moment 作为驱动目标。**

实现上不要再让 EB backbone 直接拉 2D mesh，而应把 (M_a(s,t)) 转换成自平衡的 continuum active muscle stress；同时让被动 continuum material 提供真实的软组织和纤维支撑。这样鱼体才会从“被中线拉扯的纸片”变成“会主动弯曲的柔性实体”。
