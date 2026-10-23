# R0 执行完成，等待人工验收

更新日期：2026-10-16（Asia/Shanghai）。项目根目录：`/home/yg/THIS-Lasso`。

## 当前结论

Julia 1.9.4 与独立 Python 3.11.17 环境均已配置，指定依赖安装成功，两端 smoke tests、Python 依赖完整性检查和环境激活检查均通过。源码 commit、许可证与依赖版本已记录，数据契约与学习笔记已完成。VS Code 已配置 Julia 扩展与解释器入口，图形界面内的 REPL 留待人工验收。

此报告描述最终状态。安装前的预检报告与零散安装日志已合并进 [setup_history.log](r0_logs/setup_history.log)，每段标明原文件名及 SHA-256，并保留原始命令、输出与退出码。各项验收证据见 [日志索引](r0_logs/README.md)。

## 环境状态

| 项目 | 已核查结果 | 状态 |
| --- | --- | --- |
| 系统与 Git | Linux x86_64；Git 2.34.1 | 可用 |
| Julia | `/home/yg/.local/opt/julia-1.9.4/bin/julia`，1.9.4 | 已安装、版本命令退出 0 |
| 安装包 | 清华镜像下载，SHA-256 与官方 1.9.4 校验值一致 | 已校验 |
| Julia depot | 项目 `.julia-depot-r0/`，独立于既有环境 | 已建立 |
| Julia 依赖 | Combinatorics 1.1.0、DelimitedFiles 1.9.1、EDF 0.8.1、FFTW 1.9.0、LinearAlgebra（1.9 标准库）、Statistics 1.9.0 | 安装及预编译退出 0 |
| Julia 最小测试 | 43 项断言通过；另验证 Julia 1.9 版本 | 退出 0 |
| Python | 项目 `.venv-r0/bin/python`，3.11.17 | 环境创建退出 0 |
| Python 依赖 | numpy 1.26.4、scipy 1.11.4、scikit-learn 1.6.1、statsmodels 0.14.0、pandas 2.2.3、matplotlib 3.9.4、skglm 0.5、tabulate 0.9.0；numba 0.61.2、llvmlite 0.44.0 | 安装退出 0；pip check 退出 0 |
| Python 最小测试 | LassoLarsIC、原始 Su 模块导入及原始单节点求解器基本调用 | 退出 0 |
| VS Code | Julia 扩展 1.249.2 已安装，`.vscode/settings.json` 指向新环境 | GUI REPL 待人工确认 |

Julia 安装包 SHA-256：`07d20c4c2518833e2265ca0acee15b355463361aa4efdab858dad826cf94325c`。Julia 直接依赖与完整间接依赖由 [Project.toml](../environment/r0/julia/Project.toml) 和 [Manifest.toml](../environment/r0/julia/Manifest.toml) 记录。

Python 的 [requirements.in](../environment/r0/requirements.in) 固定直接依赖，保留 Su README 指定的 statsmodels 0.14.0，并约束 NumPy/SciPy 与 skglm 的编译运行时。完整 28 个发行包版本保存在 [requirements.lock.txt](../environment/r0/requirements.lock.txt)，Python 与原生库保存在 [conda-linux-64.lock.txt](../environment/r0/conda-linux-64.lock.txt)。锁定版本已逐项与实际环境核对；Conda pip 的构建机本地地址已规范为实际版本号。没有修改既有 Python 环境或系统 Python。

## 官方源码与来源

| 仓库 | 固定 HEAD | 许可证 | 位置 |
| --- | --- | --- | --- |
| [THIS](https://github.com/TaylorBasedHypergraphInference/THIS) | `865efc43fe8ec21966659b482ca5756396974425` | GPL-3.0 | `upstream/THIS/` |
| [Su Adaptive LASSO](https://github.com/Weiwei-Su112/Adaptive-LASSO-for-higher-order-oscillatory-systems) | `fa0d662f68b7b41a1c6c3401af9504929126dafd` | MIT | `upstream/Su-AdaptiveLASSO/` |

两个 Git 仓库已成功克隆，工作区均无修改。来源、HEAD、LICENSE 哈希及已跟踪文件哈希见 [upstream_lock.json](../environment/r0/upstream_lock.json)。原始下载副本已在逐文件确认与固定上游一致后删除；该 JSON 保留历史比对结果，并注明清理状态。以后仅使用 upstream 中的两份官方源码。

## 核心接口与学习产物

| 作用与公式 | 函数入口 | 输入 → 输出 |
| --- | --- | --- |
| 完整单项式库，`P=binomial(N+dmax,dmax)` | `upstream/THIS/this-tools.jl:get_θd` | `X:[N,K]` → `θ:[P,K]`、`d:[P,dmax]` |
| 岭初始化 `Ξ=Yθᵀ(θθᵀ+ρI)⁻¹`，阈值置零并继续岭重拟合 | `upstream/THIS/this.jl:mySINDy` | θ、`Y:[N,K]` → `Ξ:[N,P]`、err |
| 特征生成、求解与结构解码 | `upstream/THIS/this.jl:this` | X,Y,ooi,dmax,λ,ρ,niter → Ainf,coeff,relerr |
| 绝对初值缩放特征、LARS/BIC、系数还原 | `upstream/Su-AdaptiveLASSO/GeneralInteraction.py:solve_ada_lasso` 与单节点版本 | 特征转置 `[K,P]`、响应 `[K]`、初始系数 → 还原后的系数 |

需要保留的源码事实：

- THIS 的 `err` 实际为 `norm(Y-Ξ*θ,1)`；`relerr=err/norm(Y,1)`，不是平方误差。
- `this` 按阶次返回 Ainf，当前函数体并未利用 `ooi` 过滤；后续评价须显式取 2、3 阶。
- 解码排除重复源节点及含目标节点的源集合，并用 `abs(coeff)>1e-8` 判边。
- Su 顶部直接导入 `skglm.GroupLasso`，与 README 的 optional 表述存在差异；当前已成功导入原始模块，包括该依赖。
- Su 的缩放为 `a_j=abs(beta_init_j)`、`X*_j=X_j*a_j`、`beta_j=a_j*theta_j`。本阶段没有实现项目自己的 Ridge 初始化 Adaptive LASSO。

正式 [数据契约](../docs/data_contract.md) 已记录数组方向、Julia 1-based 编码、常数项、重复索引、导数来源及轨迹分割原则。[R0 学习笔记](R0_learning.md) 包含公式、函数映射、手算示例和三个自查问题。

## 测试与实际命令

Julia 已执行的命令（通过 `scripts/r0_run_logged.py` 记录）：

```bash
environment/r0/julia.sh --startup-file=no --project=environment/r0/julia scripts/smoke_r0.jl
```

结果见 [julia_smoke.log](r0_logs/julia_smoke.log)：N=3、K=5 时，dmax=2 输出 θ=(10,5)、d=(10,2)，dmax=3 输出 θ=(20,5)、d=(20,3)；逐项验证单项式值、常数项、索引与原始 mySINDy/this 基本调用。共 43 项断言通过，退出码 0。

该测试的 `Y=copy(X)` 是接口夹具，不是物理导数或动力学轨迹。没有计算结构恢复指标。

Python 已执行：

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 .venv-r0/bin/python scripts/smoke_r0.py
.venv-r0/bin/python -m pip check
```

结果见 [python_smoke.log](r0_logs/python_smoke.log) 和 [python_dependency_check.log](r0_logs/python_dependency_check.log)，均退出 0：

- 固定随机种子 20261016，LassoLarsIC 输入为 80×5，BIC 选出 α=0.0056908236702308797，训练夹具 MSE=0.0019943814886417008。
- 原始 Su GeneralInteraction 模块导入成功，包含 skglm；在小型线性回归夹具上调用作者未修改的 `solve_ada_lasso_one_node`，返回有限系数，α=0。
- 该调用仅提供函数所需的三个元数据属性，没有构造振子系统或运行仿真；α=0 是本夹具的真实选择结果，不代表结构恢复结论。
- `pip check` 输出 `No broken requirements found.`。

环境激活命令 `source environment/r0/activate.sh` 已核查，python/julia 分别指向项目独立 Python 环境和新安装的 Julia 1.9.4。未计算论文性能指标，也未运行 EEG 或 Fig.3/Fig.4 实验。

## 安装过程的问题

- 默认本地代理不可用，下载最初退出 7；临时绕过代理后可联网。
- Julia 官方站直连在 300 秒超时，退出 28；清华镜像下载后通过官方 SHA-256 核验。
- Conda 离线缓存不足，离线创建退出 1；随后联网创建成功，退出 0。
- Python 官方源下载缓慢，主动停止后改用镜像；前一次子进程退出 -15，首次镜像尝试因会话中断未完成；恢复后固定依赖版本并重新安装成功，退出 0。

上述失败和重试的原始内容均保留在 setup_history.log；没有修改或补写旧尝试的退出码。最终依赖安装段为 `python_dependencies_resume.log`，环境激活段为 `activation_check.log`，均在合并历史中可检索。

## 产出与清理

R0 保留的实际产物：

| 位置 | 文件及用途 |
| --- | --- |
| 项目根目录 | README.md、.gitignore、.vscode/settings.json：项目入口、排除规则与解释器配置 |
| `environment/r0/` | README.md、activate.sh、julia.sh：环境使用与激活；requirements.in、requirements.lock.txt、conda-linux-64.lock.txt：Python 版本；julia/Project.toml、Manifest.toml：Julia 版本；upstream_lock.json：源码来源 |
| `scripts/` | r0_run_logged.py、smoke_r0.jl、smoke_r0.py：可重复执行并记录输出的最小测试 |
| `docs/` | data_contract.md：矩阵、索引、特征与导数契约 |
| `reports/` | R0_report.md、R0_learning.md：当前报告与学习笔记；r0_logs/：合并历史和关键验收日志 |
| `upstream/` | 两个固定 commit 的官方仓库，保留许可证和原始实现，工作区无修改 |

按用户要求，在两端测试通过后清理了：

- 已核查一致的两个原始下载副本及 archive 目录；确认不含唯一科研文件。
- 用户目录内的 Julia 完整下载包与失败的部分下载包。
- 一次性 environment/r0/install_julia.jl；环境重建改用保留的 Manifest 和 Pkg.instantiate。
- 合并后的 25 个历史文件原副本，原始内容与哈希保存在 setup_history.log；保留独立测试日志。

删除文件总计约 168.7 MB（十进制文件字节数，约 160.9 MiB）。详见 [cleanup.log](r0_logs/cleanup.log)。保留了已安装 Julia、Python 环境和 Julia depot，以便后续阶段直接使用；未清空全局缓存。

## 验收状态与下一阶段前提

- 环境：Julia 1.9.4、Python 3.11.17 已配置，指定依赖已安装并锁定。
- 源码：官方 HEAD、许可证及文件哈希已记录；上游未改动。
- 最小测试：Julia 43 项断言通过；Python 回归与原始 Su 基本调用通过；pip check 通过。
- 学习产物：数据契约、公式/函数/维度对照、中文学习笔记已完成。
- 人工验收：待用户在 VS Code 验证 Julia REPL，并核对日志与契约；任务文件与总进度表未自行勾选。
- R1 前提：用户人工验收 R0，并明确发出启动 R1 的新指令。

人工复查命令见 [environment/r0/README.md](../environment/r0/README.md)。本阶段未实现动力学模拟器、项目自己的 Adaptive LASSO 或跨语言融合。

R0 完成；等待人工验收；未执行下一阶段。
