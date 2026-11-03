# 重构后复现与使用

本项目保留原THIS Taylor特征和高阶解码，公共模块提取不改变科研数学路径。当前唯一科研主报告是 [R5_report.md](../reports/R5_report.md)，文档入口为 [README.md](README.md)，历史材料见 [归档索引](../archive/INDEX.md)。

本次完成状态及实际测试证据见 [S6_final_report.md](refactor/S6_final_report.md)。

## 当前环境

```bash
cd /home/yg/THIS-Lasso
.venv-r0/bin/python --version
environment/r0/julia.sh --version
```

已使用的环境为Conda prefix `.venv-r0`，Python3.11.17、NumPy1.26.4、SciPy1.11.4、scikit-learn1.6.1、matplotlib3.9.4，Julia1.9.4。环境锁和来源见 [environment/r0/README.md](../environment/r0/README.md)；没有在本次整理中安装/升级依赖。

在**新的项目副本、目标环境目录不存在时**，最小重建依据为：

```bash
conda create --prefix "$PWD/.venv-r0" --file environment/r0/conda-linux-64.lock.txt --yes
.venv-r0/bin/python -m pip install --requirement environment/r0/requirements.lock.txt
JULIA_DEPOT_PATH="$PWD/.julia-depot-r0" JULIA_PROJECT="$PWD/environment/r0/julia" julia --startup-file=no -e 'using Pkg; Pkg.instantiate()'
```

Julia二进制应先安装为1.9.4；原环境说明记录了来源和安装位置。上述原生锁面向Linux x86_64。当前包装器和审计入口沿用本机的已记录Julia路径；迁移机器需显式调整运行时位置，不能在当前目录覆盖环境或声称已在另一台机器验证重建。

## 一条命令核对现有科研基线

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python -B docs/refactor/S6_audit.py
```

该命令自动选择新时间标签，在 `docs/refactor/S6_work/` 创建独立目录，拒绝覆盖已有输出；可用 `--label 自定义唯一名称` 固定标签。它会核对源/证据哈希、执行现有单元测试、Julia/Python单trial、S1同路径Golden、六份统计表及10000次配对bootstrap索引，并检查带help的CLI。正式results只读，不重拟合30-seed，不运行R4 B或E2/E3。

原可选Python缓存清理已记入批准日志；审计既接受已删除缓存缺失，也会记录其日后从未变源文件重新生成的状态。两原论文、源码和科研数据仍按严格哈希规则检查。新审计使用单线程、禁止pycache写入、隔离临时目录与depot；不调用会写回旧manifest的历史audit。

手动只运行Python测试可使用：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-r0/bin/python -B -m unittest discover -s tests -p 'test_*.py' -v
```

原R0–R5 logger/audit有历史输出路径和源码哈希保护。S2合法改变的六个自有文件与旧源码哈希不同，旧driver可能按原规则拒绝；不要改旧manifest或覆盖正式目录来绕过。当前推荐的基线检查入口是上述S6审计。

## 六种方法与数据合同

| ID | 数学路径 | 实现入口 |
| --- | --- | --- |
| M0 | 原尺度THIS/STLSQ，官方λ=0.1、ρ=1、niter=10 | `src/solvers/stlsq.jl:fit_stlsq_raw` |
| M0s | 公共训练标准化后原生STLSQ，再还原原尺度 | `fit_stlsq_standardized`；保留R4控制 |
| M1 | 普通LASSO + guarded BIC | `src/solvers/methods.py:fit_node` |
| M1v | 普通LASSO + 独立轨迹验证 | 同上；保留R4匹配控制 |
| M2 | Ridge初值Adaptive LASSO + guarded BIC | 同上；Adaptive核心未改 |
| M3 | 同Adaptive权重 + 独立轨迹验证MSE | 同上 |

Julia X/Y为[N,K]、theta为[P,K]，Python设计矩阵为[K,P]；d节点ID为Julia1-based，0为填充，不能统一减1。外部标准化仅训练拟合，LASSO不惩罚偏置，STLSQ仍保留原常量惩罚。原尺度系数严格abs(C)>1e−8后解码，排除重复源和目标在源内，候选全集及AP定义保持。详见 [data_contract.md](data_contract.md)、[S2接口映射](refactor/S2_api_map.csv)。

## 新科研实验应如何启动

未来实验应先注册新的版本化协议、源文件清单、seed与输出目录，调用共享method/预处理/解码API，保留M0s/M1v控制；不能直接复用旧logger的默认正式目录。整理授权不自动授权新数据设计、参数搜索或新增噪声实现。当前没有为了重构而补跑科研实验。

尚未解决的科研问题：0%噪声BIC方差门槛导致M1/M2为NA；Ridge弱项初值接近0时Adaptive过度惩罚；较高噪声下M2不优于普通LASSO；M3假阳性较多。R4完整B、E2过程噪声和E3状态观测噪声仍未运行，方案在 [后续计划](r5_followup_noise_plan.md)。它们不是文件整理失败，也没有被本次重构“修好”。

## 清理与恢复

只清理6385个可选单链接Python缓存，源文件、环境锁、共享缓存、Julia原生缓存和科研记录保留。精确路径/原哈希/恢复依据见 [S5删除日志](refactor/S5_deletion_log.csv)、[清理说明](refactor/S5_report.md)。恢复工具默认只检查，不覆盖任何文件：

```bash
.venv-r0/bin/python -B docs/refactor/S5_restore_caches.py
```

旧README有完整快照；历史报告、smoke和aborted原位归档，并通过 [archive/LINKS.md](../archive/LINKS.md) 解释历史相对链接。不要清空archive或把aborted样本混入正式30-seed统计。
