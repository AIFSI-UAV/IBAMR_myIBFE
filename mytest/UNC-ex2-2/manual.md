这段代码实现了围绕固定枢轴做匀角速度旋转，旋转到指定释放时间后取消约束，并用弹簧模型把实际 Lagrangian 点拉向旋转后的目标位置。
最终计算得到的力用于 IBAMR 的流体-结构耦合（通过力分布到 Euler 网格）。
{
// pivot (example)枢轴坐标，示例中为原点
    const double Xc0 = 0.0;
    const double Xc1 = 0.0;

// rotate for 1 second, then release；t_release = 3.0：在 time >= t_release 时停止施加旋转约束。
    const double t_release = 3.0;

    libMesh::Point X_target;
if (time >= t_release)：释放阶段，设置 X_target = x，意味着目标位置等于当前实际位置，从而使得后续计算的弹簧力为零（或接近零）。
    if (time >= t_release)
    {
        // release: F=0
        X_target(0) = x(0);
        X_target(1) = x(1);
    }
    else
    {
        const double omega = 0.5 * M_PI;
        const double theta = omega * time;

        const double c = std::cos(theta);
        const double s = std::sin(theta);

        const double dX0 = X(0) - Xc0;
        const double dX1 = X(1) - Xc1;

        X_target(0) = Xc0 + c * dX0 - s * dX1;
        X_target(1) = Xc1 + s * dX0 + c * dX1;
    }

    F = kappa_s * (X_target - x);
}

振动阶段：点在 y 方向上做简谐运动。

振动条件 (oscillation)  

在未释放之前，目标位置在 y 方向上会进行简谐振动：
ytarget=y+A⋅sin⁡(ωt+ϕ)