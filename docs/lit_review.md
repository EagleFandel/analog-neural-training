# 文献调研与连续时间推导（提要）

## 1. Neural ODE 与连续优化动力学
- Chen et al., 2018, Neural Ordinary Differential Equations：将残差网络极限视为连续深度，提出 ODEBlock 与自适应求解；伴随法实现内存友好的梯度计算。
- Su, Boyd, Candes, 2016：优化视为连续时间系统，分析 Nesterov/Heavy-ball 对应的二阶动力学与收敛性质。

## 2. 保结构与辛积分
- Hairer, Lubich, Wanner, Geometric Numerical Integration：辛积分保持哈密顿结构与近似能量不变性，在长期模拟上优势显著。
- 在优化动力学中用于动量法的几何稳定性与更大步长下的轨迹平稳性。

## 3. 隐式/IMEX 与刚性问题
- Ascher et al., 1995：IMEX 方法在刚性系统中的稳定性优势；对强凸或高条件数任务更友好。
- 线性化/CG 求解可扩展，结合预条件降低迭代次数。

## 4. 自然梯度与镜像流
- Amari, 1998：自然梯度以 Fisher 度量调整下降方向，改善病态问题条件数。
- Wibisono et al., 2016：Bregman Lagrangian；镜像流与加速方法的连续时间解释。

## 5. 噪声与泛化（SDE/SGD）
- Mandt et al., 2017：SGD 近似为常微分方程上的 Ornstein-Uhlenbeck 过程；噪声与平坦极小值的关系。
- 课程与综述：SGD 噪声对泛化的影响与可视化直观。

## 6. 模拟/混合计算与类脑芯片
- Memristor crossbar、模拟-数字混合加速器综述：以阵列并行与低精度实现矩阵运算；
- 结合连续时间训练框架，讨论硬件映射与能耗潜力。

## 7. 连续时间推导要点
- 梯度流 ODE：\(\dot{\theta}=-\nabla L(\theta)\)；PL 条件下指数收敛；
- 动量 ODE：\(\ddot{\theta}+\gamma\dot{\theta}+\nabla L=0\)；辛积分降低能量漂移；
- IMEX：\((I+h\nabla^2 g)\theta_{k+1}=\theta_k-h\nabla f(\theta_k)\)；
- 自然梯度/镜像：\(\dot{\theta}=-G^{-1}\nabla L\)，或 \(\tfrac{d}{dt}\nabla\psi(\theta)=-\nabla L\)；
- SDE：\(d\theta=-\nabla L\,dt+\sigma dB_t\)，稳态误差与泛化权衡。

> 本调研为写作与答辩提供引用脉络与连续时间推导依据，详证见 `docs/theory.md`。

