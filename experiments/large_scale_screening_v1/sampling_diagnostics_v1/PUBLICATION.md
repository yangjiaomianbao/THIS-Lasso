# sampling_diagnostics_v1：GitHub 发布范围

本次推送包含独立采样/轨迹对照实现、测试、审计协议、复现命令，以及 N15/N30 各30个配对拓扑的主实验和两种子 B1–B4/追加激励控制结果。主实验27,000个拟合全部通过原BIC；总计29,700个拟合中28,440 OK、1,260 NA。没有放宽BIC门槛或更改旧神经权重。

- [综合报告与 Q1–Q3 结论](../../../results/large_scale_screening_v1/sampling_diagnostics_v1/REPORT.md)
- [本次推送文件及哈希](../../../results/large_scale_screening_v1/sampling_diagnostics_v1/github_published_files.csv)
- [完整本地产物清单及上传标记](../../../results/large_scale_screening_v1/sampling_diagnostics_v1/github_artifact_inventory.csv)
- [发布核验回执](../../../results/large_scale_screening_v1/sampling_diagnostics_v1/github_publication.json)

## 已上传

完整实验代码、配置/种子协议、源码快照、66项测试日志及独立复算回执；所有逐目标/逐网络 JSON 和 CSV 指标、选参锁记录、模型系数哈希、展开中心与相位盲区系数、数据元信息、完整任务校验清单；核心对照/消融表、奇异值谱与中心/尺度敏感性、PNG/PDF诊断图。诊断级的 matrix_singular_values.npz 和 full_quadratic_singular_values.npz 同样上传。

## 保留本地的数组

完整本地产物约908 MiB（含文件系统块开销）。以下三个集合约755 MiB的文件内容未放入Git历史，原文件均保留：

- `jobs/*/arrays.npz`：全部逐模型候选、评分、系数路径及奇异值，约502 MiB。
- `jobs/*/data/data.npz`：原始相位、解析导数、图与拆分数组，约101 MiB。
- `jobs/*/selection_locks/*.npz`：逐模型系数与训练尺度数组，约152 MiB。

`github_artifact_inventory.csv` 的 `published=false` 表示仅在本地。某文件出现在原 delivery_manifest 或任务 completion 的哈希表中，并不表示它已上传。REPORT 中关于这些大数组的保存说明指完整本地实验目录；GitHub 克隆只包含上述公开交付范围。

所有状态/参数/指标表和选参 JSON 均保留；公开 selection_locks.json 记录选择参数、KKT/条件数和系数SHA-256，但不等同于全部原始系数数组。科研原始目录和冻结交付清单不因发布而改写。

## 复现

按本目录 [README](README.md) 的命令使用一个新的输出目录，从固定参数与随机流重新生成全部数据和数组。不要在只含公开文件的目录直接运行 `verify`：该命令要求完整的逐任务数组，缺失时会正常拒绝；也不要用 `--out` 指向这份冻结发布目录进行补写。原配套依赖说明见仓库 environment/r0，实际本轮版本见结果中的 runtime.json。

GitHub 发布清单单独记录新增发布说明和导航；原 science_sources 与 delivery_manifest 保持原样，继续描述实验执行时的科学源码和完整本地证据。
