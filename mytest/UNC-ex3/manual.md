该代码实现了对参考坐标系下的刚性/弹性翼段做“升降（heave）+俯仰（pitch）”的运动学驱动，计算目标位置 X_target 并用弹簧系数 kappa_s 产生牵引力 F，以在 IBAMR 中把 Lagrangian 点拉向该运动轨迹。
{
    // --------------------------------------------------
    // Reference rotation point X_ref (in reference coordinates)
    // Example: quarter-chord point or hinge point
    // --------------------------------------------------
    const double Xref0 = 0.25;   // x-coordinate of rotation reference point
    const double Xref1 = 0.0;    // y-coordinate of rotation reference point
参考旋转点（Pivot）
    Xref0 = 0.25, Xref1 = 0.0：以参考坐标系的四分之一弦长点或铰链点为旋转中心。

    // --------------------------------------------------
    // Heave motion: translation of the reference point
    // Example: sinusoidal motion in y
    // --------------------------------------------------
    const double Ah = 0.1;       // heave amplitude
    const double fh = 1.0;       // heave frequency
    const double omegah = 2.0 * M_PI * fh;
    const double phih = 0.0;     // heave phase
升降（Heave）平移
    振幅 Ah = 0.1，频率 fh = 1.0，角频率 omegah = 2πfh。
    参考点在 y 方向的瞬时位置：xc1 = Xref1 + Ah * sin(omegah * time + phih)。

    const double xc0 = Xref0;    // no x-translation in this example
    const double xc1 = Xref1 + Ah * std::sin(omegah * time + phih);

    // --------------------------------------------------
    // Pitch motion: rotation angle
    俯仰（Pitch）旋转
    俯仰幅度 theta0 = 15°，频率 fp = 1.0，相位 phip = π/2。
    旋转角：theta = theta0 * sin(omegap * time + phip)；用 cos(theta)、sin(theta) 构造二维旋转矩阵。
    // --------------------------------------------------
    const double theta0 = 15.0 * M_PI / 180.0;   // pitch amplitude (15 deg)
    const double fp = 1.0;                       // pitch frequency
    const double omegap = 2.0 * M_PI * fp;
    const double phip = 0.5 * M_PI;             // phase difference w.r.t. heave

    const double theta = theta0 * std::sin(omegap * time + phip);

    const double c = std::cos(theta);
    const double s = std::sin(theta);

    // --------------------------------------------------
    // Relative position in reference configuration
    // --------------------------------------------------
    const double dX0 = X(0) - Xref0;
    const double dX1 = X(1) - Xref1;

    // --------------------------------------------------
    // Target position = translated pivot + rotated relative vector
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