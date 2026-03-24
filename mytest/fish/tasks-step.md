你的拆解方向是对的，而且**很适合现在这种 IBFE/free-swimming 调试阶段**。

这个版本和你原来的思想一致，只是把“全自由”再往后压了一步。

我建议你**保留这个总体顺序**，只做两点强化：

第一，**把“验证目标”写死到每一步**，不要一边调一边换目标。
第二，**把 full free-swimming 放得更靠后**，先把 constrained / tethered / partial-free 模式跑顺。文献上，tethered/fixed-body 本来就是鱼类推进数值中的常见起点；而 free-swimming 中的 recoil 会明显改变速度、效率和力学解释，所以确实不适合一开始就全自由。([AIP Publishing][1])

## 我建议的最终拆解顺序

### 0. 纯运动学审计

在真正耦合前，单独检查 `compute_eel_target()`。

你要看的是：

* 选 5–7 个固定的 `s/L`
* 直接输出 `xtar, ytar, utar_x, utar_y`
* 和解析目标波形逐点比对

通过标准：

* 头到尾相位单调后移
* 振幅单调增大
* `tail_y` 的峰峰值与设定包络一致
* `body_wave_v()` 与数值差分速度一致

这一步不需要流体正确，只需要**目标函数正确**。

---

### 1. target-only，固定 (x,y,\theta)

这一步你的想法完全对。

目的不是看游动，而是看：

* 形变是否能跟踪 target
* 头小尾大的 traveling wave 是否正确
* penalty 参数会不会导致局部抖动或过冲

建议输出：

* 5 个 body stations 的 `y(s,t)`
* 尾端 `tail_y(t)`
* `L2` 跟踪误差：(|x-x_{tar}|)

通过标准：

* 目标波形形状对
* 跟踪误差小
* 没有非物理高频振荡

---

### 2. target-only，固定姿态下的小振幅

这一步非常必要。

原因很简单：大振幅时你分不清问题来自

* 目标函数错
* penalty 太硬/太软
* 还是几何非线性/耦合太强

所以先用小振幅把“数值基础链路”跑通，再放大。这个做法和很多 fish locomotion / self-propelled 文献里“先从 prescribed kinematics 入手，再进入 fully coupled free swimming”的建模习惯是一致的。([politesi.polimi.it][2])

建议：

* 先把当前振幅降到 30–50%
* 周期先跑 3–5 个
* 先只看波形与误差，不看速度

---

### 3. target-only，只放开 (x) 平移

这一步我很赞同，而且我建议你把它明确叫做：

**“net thrust test under constrained attitude”**

目的：

* 不让偏航和横漂污染判断
* 先回答最基本的问题：**这套体波有没有净推进趋势**

建议输出：

* `x_com(t)`
* 一个周期平均的 `u_com`
* 周期平均流向力或等效推进量
* `tail_y(t)`

通过标准：

* 有稳定的净前进趋势
* 不出现反复前后拉扯但平均速度接近 0 的情况
* 参数变化时推进趋势有可解释响应

这一步很关键，因为 tethered / constrained 与 self-propelled 的结果本来就可能不同，你现在这样分开看是合理的。([科学直接][3])

---

### 4. target-only，再放开 (\theta)，最后才放开 (y)

我同意你这个顺序，而且建议比你写得更明确一点：

#### 4a. 放开 (x,\theta)，锁死 (y)

这一步专门看：

* 自发偏航是否过强
* `theta(t)` 是否连续
* 推进是否因为角漂移被破坏

#### 4b. 再放开 (x,y,\theta)

这时才是真正的 full free-swimming target-only baseline。

原因是：

* angular recoil 对效率和推进表现很重要
* lateral recoil 也会改变结果
* 但一上来全放开，很难判断是哪个自由度先出问题。([PMC][4])

---

### 5. active-only，先固定姿态

这一步也对。

我建议把目标写得更窄一点：

**先验证 active stress 能不能产生正确方向、正确相位、正确分布的体内驱动力。**

先不要追求“会游”。
先看：

* 应力场是否只出现在预期区域
* ramp 后形变方向是否正确
* 尾部是否比头部响应更明显
* 不会一开 active stress 就爆解

你说的这几点我都同意：

* 关掉 target penalty
* `ACTIVE_STRESS_T0` 不要和 `KAPPA_S_BODY` 绑死
* `ACTIVE_RAMP_TIME` 至少一个周期，不要 0

---

### 6. active-only，只放开 (x)

这一步和 target-only 的第 3 步一一对应。

这样你就能直接比较：

* 同样的波输入下
  **target-only 的净推进**
  vs
  **active-only 的净推进**

这一步会非常有价值，因为它能告诉你：

* 你现在 active model 是不是“只会局部抖动”
* 还是已经具备产生净推进的能力

---

### 7. active-only，再逐步放开 (\theta)、(y)

顺序仍然建议：

* (x) only
* (x+\theta)
* (x+y+\theta)

这样最稳。

---

## 我建议你再加的两个“硬规则”

### 规则 A：每一步只改一件事

例如从第 3 步到第 4 步，只改：

* 从 (x)-free 变成 (x+\theta)-free

不要同时改：

* 振幅
* penalty
* dt
* active 参数

否则你会完全失去定位能力。

---

### 规则 B：每一步都设“通过标准”

例如：

#### Step 1 通过标准

* `tail_y(t)` 周期稳定
* 5 个 body stations 相位正确
* 跟踪误差低于阈值

#### Step 3 通过标准

* 平均 `u_com > 0`
* `theta` 不发散
* 周期平均推进量符号稳定

#### Step 5 通过标准

* active stress 作用区域正确
* ramp 后无数值爆振
* 位移方向与预期一致

这样你会快很多。

---

## 我对你当前顺序的一个小调整

我会把你的顺序微调成这样：

1. `compute_eel_target()` 纯运动学审计
2. target-only，固定 (x,y,\theta)
3. target-only，小振幅
4. target-only，只放开 (x)
5. target-only，放开 (x+\theta)
6. target-only，全自由
7. active-only，固定 (x,y,\theta)
8. active-only，只放开 (x)
9. active-only，放开 (x+\theta)
10. active-only，全自由

这个版本和你原来的思想一致，只是把“全自由”再往后压了一步。

---

## 最后一句判断

**你的拆解思路是对的，而且比“一上来全自由 self-propulsion”更稳。**
最重要的是：
先把 **target-only baseline** 做成一个可靠的控制组，再切到 **active-only**。因为 free-swimming 中 recoil 会显著改变结果，而 constrained / tethered 与 self-propelled 结果本来就不一样，所以你现在这种逐步放开自由度的策略是很合理的。([科学直接][3])

如果你愿意，我下一步可以把这套拆解直接整理成一份 **“调试流程 LaTeX 笔记”**。

[1]: https://pubs.aip.org/aip/adv/article/14/10/105111/3315841/Kinematics-and-hydrodynamic-performance-of?utm_source=chatgpt.com "Kinematics and hydrodynamic performance of zebrafish C- ..."
[2]: https://www.politesi.polimi.it/retrieve/cc8861da-3ede-429d-8f64-f0b4efd8c1cf/2023_12_Murakami_Thesis_01.pdf?utm_source=chatgpt.com "An Immersed Boundary method for fish-like swimming ..."
[3]: https://www.sciencedirect.com/science/article/abs/pii/S0029801822017541?utm_source=chatgpt.com "A comparison for hydrodynamic performance of undulating ..."
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10526200/?utm_source=chatgpt.com "How Free Swimming Fosters the Locomotion of a Purely ..."

