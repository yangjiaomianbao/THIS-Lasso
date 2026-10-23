# GitHub 上传范围与使用说明

2026-10-23，目标仓库：<https://github.com/yangjiaomianbao/THIS-Lasso>。

本次按“代码、配置、文档和正式结果表，排除大型中间数据”准备。它是源码与结果展示版本，不是本机完整实验目录的备份。

## 结果入口

- [15节点、五类动力学实验报告](../results/dynamics15/five_systems_certified_20261022/REPORT.md)。
- [三体结构恢复结果表](../results/dynamics15/five_systems_certified_20261022/tables/table1_threebody.md)。
- [Excel结果表](../results/dynamics15/five_systems_certified_20261022/tables/table1_threebody.xlsx)。
- [三体指标核验](../results/dynamics15/five_systems_certified_20261022/tables/metrics_validation.md)。
- [R5正式报告](../reports/R5_report.md)及[逐种子指标](../results/r5/e1_full_independent/trials.csv)。

## 纳入 Git

- `src/`、`scripts/`、`tests/`、`configs/`。
- `environment/` 中的版本锁定文件及环境重建说明。
- 文档、报告、任务说明和维护记录，排除历史工作副本及压缩备份。
- R5正式批次根目录的原始指标、弱边诊断、统计结果、实验配置与模型选择锁。
- 15节点正式批次根目录的连续边得分、原始指标、配置、模型选择锁、验证记录和图表。
- 上述正式批次中存在的 `tables/`、`logs/`、`source_snapshot/`、图表目录。

## 仅保留在本机

- Python环境 `.venv-r0/`、Julia包缓存 `.julia-depot-r0/`、编辑器设置和编译缓存。
- `data/`、实验子目录中的逐节点模型文件和大型中间矩阵（例如 `theta.npy`、`Z_train.npy`）。
- smoke、历史对照、中止批次等未列入本次正式结果展示范围的实验数据。
- `archive/`、`docs/refactor/S6_work/` 和文档中的压缩备份。
- `upstream/` 的两个独立Git克隆；恢复方式见下文。

所有排除项均保留在原本机目录，上传操作不删除或改写它们。报告和完整性清单按原样保存，其中部分链接指向本次未上传的文件；这些历史清单描述完整本地实验，不表示GitHub副本含有清单中的全部文件。

## 恢复上游源码

在项目根目录执行，且目标路径应尚不存在：

```bash
git clone https://github.com/TaylorBasedHypergraphInference/THIS.git upstream/THIS
git -C upstream/THIS checkout 865efc43fe8ec21966659b482ca5756396974425
git clone https://github.com/Weiwei-Su112/Adaptive-LASSO-for-higher-order-oscillatory-systems.git upstream/Su-AdaptiveLASSO
git -C upstream/Su-AdaptiveLASSO checkout fa0d662f68b7b41a1c6c3401af9504929126dafd
```

版本和原始许可证记录见 [upstream_lock.json](../environment/r0/upstream_lock.json)。环境重建见 [environment/r0/README.md](../environment/r0/README.md)。部分入口包含原工作站的Julia路径，新机器需配置对应运行环境。

当前实验入口有严格的历史交付完整性检查。由于本次没有上传全部历史数据，仅克隆此仓库并不满足这些检查；若要按原入口完整复现，须从本机备份恢复依赖数据。不能把缺失数据解释为实验失败，也不应为了运行而静默绕过完整性检查。

## 本地 Git 忽略配置

原 `.gitignore` 属于已有审计基线，本次保持原样。额外排除规则写入本机 `.git/info/exclude`，正式结果文件按明确清单加入暂存区；已跟踪文件后续修改仍可正常提交。

新克隆如要采用相同本地规则，可将以下内容追加到 `.git/info/exclude`：

```gitignore
/.vscode/
/archive/
/data/
/results/
/docs/refactor/S6_work/
*.tar.gz
*.zip
.env
.env.*
*.pem
*.p12
```

根README和历史报告中的“尚未初始化Git”等语句属于原审计时点的记录，本次没有改写历史文件；Git上传范围以本文件为准。
