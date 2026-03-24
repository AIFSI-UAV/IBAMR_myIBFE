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