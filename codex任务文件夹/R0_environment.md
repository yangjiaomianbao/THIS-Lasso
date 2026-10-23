# R0 — 环境配置、源码核验与接口学习

**启动条件**：独立阶段，无需先运行其他阶段。
**本阶段目标**：在 VS Code 中建立 Julia 1.9 + Python 3.11 环境，确认官方 THIS 和 Su 仓库核心功能可调用，掌握公式与数据维度。**不生成动力学数据、不实现新回归、不融合两种语言**。

## 复制给 Codex 的独立指令

> 你现在只执行 `R0_environment.md`，不得实施 R1–R5。首先检查操作系统、VS Code 当前工作区、Julia/Python/Git 是否可用，以及仓库是否已经存在。报告检查结果和拟修改文件清单，再在 R0 范围内执行。若系统级软件或权限不足，不用 sudo、不改变系统环境，给出手动安装步骤并停止等待用户。不要运行全量 EEG 或 Su Fig.3/Fig.4 耗时实验。
>
> 1. 在 `upstream/THIS` 和 `upstream/Su-AdaptiveLASSO` 仅当尚不存在时克隆两个官方仓库，记录 `git rev-parse HEAD`、仓库地址、许可证。上游文件作为只读参照；`upstream/` 应加入 .gitignore，或者说明采用 git submodule 的方案，不擅自提交嵌套源码。
> 2. 阅读 THIS 官方 README，核对 Julia 1.9 及依赖 `Combinatorics`, `DelimitedFiles`, `EDF`, `FFTW`, `LinearAlgebra`, `Statistics`。基于实际环境安装可用的包。阅读 Su README，创建 Python 3.11 环境；只安装本阶段需要的依赖，不为可选扩展强行大改代码。特别检查 `GeneralInteraction.py` 文件顶部是否直接导入 `skglm`，并记录 README 'optional' 与导入行为之间是否存在差异。
> 3. 阅读 `upstream/THIS/this-tools.jl` 里的 `get_θd`、`this.jl` 里的 `mySINDy` 和 `this`，阅读 Su `GeneralInteraction.py` 中与 Adaptive LASSO、LARS+BIC 相关的代码。按“论文公式 → 代码函数 → 输入/输出维度 → 作用”建立对照表。不要将 Su 的正弦耦合特征写入 THIS。
> 4. 写最小功能 smoke tests：Julia `include("this.jl")`、`get_θd` 在简单 3 节点数据上输出正确维度；Python 能导入 `LassoLarsIC` 并执行小型回归。仅测试基本调用，不要求复现论文图。
> 5. 创建 `docs/data_contract.md`，锁定原始 THIS 的 `X,Y:[N,K]`, `θ:[P,K]`, `d:[P,dmax]`, `Ξ:[N,P]`，Python 用 `θ.T:[K,P]`；解释 Julia 1-based 索引、常数项、重复变量单项式、解析导数和估计导数的区别。
> 6. 输出 `reports/R0_report.md`：安装/运行命令及结果、上游 commit、依赖版本、核心函数导图、环境问题、下一阶段所需前提。并在 `reports/R0_learning.md` 用简单例子向我解释 THIS 是怎么通过 Taylor 特征变为线性回归的，最后提出最多三个思考题。
> 7. R0 完成后立即停止，不写模拟器，不写 Adaptive LASSO，不修改 `this.jl`。

## 本阶段允许新建或修改的文件

`docs/data_contract.md`, `reports/R0_report.md`, `reports/R0_learning.md`, `scripts/smoke_r0.jl`, `scripts/smoke_r0.py`, `.gitignore`, R0 环境说明/依赖锁定文件。`upstream/` 内禁止代码改动。

## 手动验收建议

在 VS Code 终端分别验证 `julia --version`、`python --version`；运行 Julia 与 Python smoke test；检查 `get_θd` 返回的行列方向。确认文档里的公式和 Julia 真实数组布局一致。

## R0 研究进度检查表

- [ ] 已配置 Julia 1.9，且 VS Code 可打开 Julia REPL。
- [ ] 已配置 Python 3.11，并确认 `LassoLarsIC` 可调用。
- [ ] 两个官方仓库已获取，HEAD commit 和许可证已记录。
- [ ] 已读懂 `get_θd → mySINDy → this` 的输入输出及索引。
- [ ] 已完成两个最小 smoke tests，真实日志可检查。
- [ ] 已形成 `docs/data_contract.md` 和 R0 学习笔记。
- [ ] **人工验收通过**，明确授权开始 R1（Codex 不得自行推进）。

**本阶段可向 Codex 提问：** ① `θ` 的维度为什么是 P×K？② `d` 中重复变量索引代表什么？③ `mySINDy` 的 λ 与 Adaptive LASSO 的 α 相同吗？
