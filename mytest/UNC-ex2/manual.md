这是上下摆动的往复运动

if (t_release > 0.0 && time >= t_release)
{
    // release: force off
    X_target(0) = x(0);
    X_target(1) = x(1);
}
else
{
    // Oscillation in y: y += A sin(omega t + phi)
    X_target(0) = X(0);
    X_target(1) = X(1) + A * std::sin(omega * time + phi);
}

振动阶段：点在 y 方向上做简谐运动。

振动条件 (oscillation)  

在未释放之前，目标位置在 y 方向上会进行简谐振动：
ytarget=y+A⋅sin⁡(ωt+ϕ)