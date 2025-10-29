# 连续时间训练动力学的理论分析草案

## 1. 梯度流与 Lyapunov 函数

考虑损失函数 \(L: \mathbb{R}^d \to \mathbb{R}\)，假设满足：

1. \(L\) 下界有限，且在最优解 \(\theta^*\) 处取得最小值；
2. 梯度 \(\nabla L\) \(L\)-Lipschitz：\(\|\nabla L(x)-\nabla L(y)\|\le L_f\|x-y\|\)；
3. 满足 Polyak–Łojasiewicz (PL) 条件：\(\frac{1}{2}\|\nabla L(\theta)\|^2 \ge \mu (L(\theta)-L^*)\)。

梯度流 ODE：
\[
\dot{\theta}(t) = -\nabla L(\theta(t)).
\]

取 Lyapunov 函数 \(V(t)=L(\theta(t))-L^*\)。沿着解轨迹求导：
\[
\dot V(t) = \langle \nabla L(\theta), \dot{\theta} \rangle = -\|\nabla L(\theta)\|^2 \le -2\mu V(t).
\]
使用 Grönwall 不等式，可得指数衰减：
\[
V(t) \le V(0)\, e^{-2\mu t}.
\]

因此连续梯度流在 PL 条件下具有指数收敛率，其时间常数由 \(\mu\) 决定。

## 2. 离散积分器的稳定性与误差界

### 2.1 显式 Euler / 高阶 Runge–Kutta

离散更新：\(\theta_{k+1} = \theta_k - h \nabla L(\theta_k)\)。局部截断误差 \(\mathcal{O}(h^2)\)，全局误差 \(\mathcal{O}(h)\)。在 PL 条件下，需满足步长上界 \(h < 2/L_f\) 方可保证单调下降。高阶 RK 方法以额外函数评估换取 \(\mathcal{O}(h^p)\) 的局部误差，对于给定目标误差 \(\varepsilon\)，所需步数 \(N \approx \mathcal{O}(\varepsilon^{-1/p})\)。

### 2.2 自适应 DOPRI54 与容差–能耗权衡

Dormand–Prince(5,4) 嵌入法通过估计四、五阶解的差值控制误差，其自适应策略保证局部误差 \(\delta\) 满足给定容差 \(\tau\)。在 Lipschitz 向量场下，NFE（函数评估次数）约为 \(\mathcal{O}(\tau^{-1/p})\)。结合 PL 条件下的指数衰减，可得达到 \(V(T)\le \varepsilon\) 的总体 NFE 上界：
\[
\text{NFE} \lesssim C \cdot \left(\kappa^{\alpha} \log\frac{V(0)}{\varepsilon}\right) \tau^{-\beta},
\]
其中 \(\kappa=L_f/\mu\) 为“连续条件数”，\(\alpha,\beta\) 依赖于容差控制策略。在实验中可拟合经验指数用于能耗预测。

## 3. 重球 ODE 与辛积分能量界

重球（Heavy-ball）ODE：
\[
\ddot{\theta} + \gamma \dot{\theta} + \nabla L(\theta)=0.
\]
定义广义动量 \(v=\dot{\theta}\)，系统可写成哈密顿形式（忽略阻尼项）：
\[
\dot{\theta}=v, \qquad \dot{v}=-\nabla L(\theta).
\]

辛积分器（如分裂式 Störmer-Verlet）对无阻尼系统保持辛结构，满足离散能量漂移界：
\[
|H(\theta_k,v_k) - H(\theta_0,v_0)| \le C h^2, \qquad H= L(\theta)+\tfrac12 \|v\|^2.
\]
加入阻尼后，通过指数因子修正可保持能量单调下降，并显著降低数值耗散与振荡。对比传统 RK4，在较大步长下辛法提供更好的轨迹平稳性与能量一致性，实验中可通过相对能量漂移量化：
\[
\Delta H_{\text{rel}} = \frac{H_T - H_0}{|H_0|}.
\]

## 4. IMEX 半隐式方法的稳定域

目标损失拆解为 \(L = f + g\)，其中 \(f\) 光滑且梯度 Lipschitz，\(g\) 表示刚性项（如强凸二次项/正则项）。IMEX 更新：
\[
\frac{\theta_{k+1}-\theta_k}{h} = -\nabla f(\theta_k) - \nabla g(\theta_{k+1}).
\]
对强凸二次 \(g(\theta)=\tfrac12\theta^T Q\theta\)，更新可化为线性方程 \((I + h Q)\theta_{k+1} = \theta_k - h \nabla f(\theta_k)\)。其稳定域包含左半平面（A-stable），相比显式法允许更大步长。使用共轭梯度解线性系统时，可通过残差 \(\|r\|\) 控制隐式误差，对条件数 \(\kappa(Q)\) 高的刚性问题具有显著优势。

## 5. 自然梯度与镜像流

自然梯度 ODE：
\[
\dot{\theta} = - G(\theta)^{-1}\nabla L(\theta),
\]
其中 \(G(\theta)\) 为 Fisher 信息矩阵或对角近似。若 \(G\) 在谱上有界且保持正定，则可证明 Lyapunov 函数 \(V=L-L^*\) 仍满足指数衰减，速度与 \(G\) 的条件数相关。镜像流（Bregman Lagrangian）满足：
\[
\frac{d}{dt}\nabla \psi(\theta) = -\nabla L(\theta),
\]
对熵镜像可解释为在概率单纯形内的自然梯度。通过能量函数 \(E(t)=L(\theta)-L^* + D_\psi(\theta^*,\theta)\) 可建立收敛界，其中 \(D_\psi\) 为 Bregman 散度。

## 6. 噪声/量化下的连续随机动力系统

加入模拟噪声与量化误差，可建模为 SDE：
\[
d\theta_t = -\nabla L(\theta_t) dt + \sigma dB_t + dQ_t,
\]
其中 \(dQ_t\) 表示量化导致的跳跃项，近似为零均值噪声。对小噪声极限 \(\sigma \to 0\)，弱误差阶数为 1，Euler–Maruyama 误差为 \(\mathcal{O}(h^{1/2})\)。稳定性分析表明：

- 若噪声满足 \(\mathbb{E}[\zeta_t]=0\)，方差 \(\propto \sigma^2\)，则期望损失满足 \(\mathbb{E}[V(t)] \le V(0) e^{-2\mu t} + \frac{\sigma^2}{2\mu}\)。
- 量化误差可被视为有界噪声，若步长与量化步幅匹配，可保持收敛；否则会在邻域 \(\mathcal{O}(\text{quantization step})\) 内波动。

## 7. 总结

上述分析提供了各类连续时间训练方案的收敛率、稳定性与能耗估计理论依据：

- 梯度流在 PL 条件下指数收敛；
- 高阶/自适应积分器通过控制步长或容差，平衡精度与函数评估开销；
- 辛积分保障能量几何，适合动量型 ODE；
- IMEX 扩展稳定域，适用于刚性与正则化问题；
- 自然梯度与镜像流引入几何预条件，改善病态条件；
- 噪声/量化可视为 SDE，给出期望收敛界与稳态误差。

后续将在论文正文中补充正式定理、证明细节与与实验结果对应的讨论。



