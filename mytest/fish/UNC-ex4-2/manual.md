该代码实现了对参考坐标系下的刚性/弹性翼段做“升降（heave）+俯仰（pitch）”的运动学驱动，计算目标位置 X_target 并用弹簧系数 kappa_s 产生牵引力 F，以在 IBAMR 中把 Lagrangian 点拉向该运动轨迹。
{
    // --------------------------------------------------
    // Reference point in the body
    // --------------------------------------------------
    const double Xref0 = 0.25;
    const double Xref1 = 0.0;

    // --------------------------------------------------
    // Circular trajectory of the reference point
    // x_c(t) = (xc(t), yc(t))
    // --------------------------------------------------
    const double Rc = 0.5;      // circle radius
    const double omega = 1.0;   // angular speed of path motion

    const double xc0 = Rc * std::cos(omega * time);
    const double xc1 = Rc * std::sin(omega * time);

    // --------------------------------------------------
    // Tangent direction of the path
    // dx/dt = -Rc*omega*sin(omega t)
    // dy/dt =  Rc*omega*cos(omega t)
    // Heading angle follows tangent
    // --------------------------------------------------
    const double vx = -Rc * omega * std::sin(omega * time);
    const double vy =  Rc * omega * std::cos(omega * time);

    const double theta = std::atan2(vy, vx);

    const double c = std::cos(theta);
    const double s = std::sin(theta);

    // --------------------------------------------------
    // Relative position in reference configuration
    // --------------------------------------------------
    const double dX0 = X(0) - Xref0;
    const double dX1 = X(1) - Xref1;

    // --------------------------------------------------
    // Target position
    // --------------------------------------------------
    libMesh::Point X_target;
    X_target(0) = xc0 + c * dX0 - s * dX1;
    X_target(1) = xc1 + s * dX0 + c * dX1;

    // --------------------------------------------------
    // Tether force
    // --------------------------------------------------
    F = kappa_s * (X_target - x);
}

振动阶段：点在 y 方向上做简谐运动。

振动条件 (oscillation)  

在未释放之前，目标位置在 y 方向上会进行简谐振动：
ytarget=y+A⋅sin⁡(ωt+ϕ)