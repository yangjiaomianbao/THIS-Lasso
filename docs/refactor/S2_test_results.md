# S2实际测试与回归记录

日期2026-10-19，Asia/Shanghai。命令由 [S2_audit.py](S2_audit.py) 记录；临时目录、depot和日志均限定在S2审计范围。Python禁写pycache，BLAS/Julia单线程；Julia禁历史和持久编译模块，旧环境保持不变。

## 最终通过的检查

| 检查 | 最终结果 | 证据 |
| --- | --- | --- |
| A批Python | 91通过 | [S2_A_retry2_python_tests.log](S2_A_retry2_python_tests.log) |
| A批Golden | 全通过，直接批量系数差0，零噪声NA保留 | [S2_A_retry2_golden.log](S2_A_retry2_golden.log) |
| A批完整N8跨语言trial | 15项断言通过 | [S2_A_retry2_trial_smoke.log](S2_A_retry2_trial_smoke.log) |
| B批Python | 95通过 | [S2_B_python_tests.log](S2_B_python_tests.log) |
| B批Golden | 全通过，直接批量系数差0 | [S2_B_golden.log](S2_B_golden.log) |
| B批完整N8跨语言trial | 15项断言通过 | [S2_B_trial_smoke.log](S2_B_trial_smoke.log) |
| C批Python | 95通过 | [S2_C_python_tests.log](S2_C_python_tests.log) |
| C批Golden | 全通过，直接批量系数差0 | [S2_C_golden.log](S2_C_golden.log) |
| C批完整N8跨语言及标准化/STLSQ | 21项断言通过 | [S2_C_trial_smoke.log](S2_C_trial_smoke.log) |
| 已有R1 Julia | 68项通过 | [S2_test_r1_baseline.log](S2_test_r1_baseline.log) |
| 已有R1 revision Julia | 27项通过 | [S2_test_r1_revision.log](S2_test_r1_revision.log) |
| 已有R3 Julia含Python子进程 | 63项通过 | [S2_test_r3_adapter.log](S2_test_r3_adapter.log) |
| 旧R3/R4/R5 CLI | 三个help均退出0 | [R3](S2_r3_cli_help.log)、[R4](S2_r4_cli_help.log)、[R5](S2_r5_cli_help.log) |
| 修改/新增Julia文件语法 | 4个文件通过 | [S2_julia_entrypoint_syntax.log](S2_julia_entrypoint_syntax.log) |
| 原文件、论文、正式数据完整性 | 仅6个预定源码变化；其余31473原条目、全部正式4113文件不变 | [S2_integrity.log](S2_integrity.log)、[JSON](S2_integrity_check.json) |

最终Python测试总数95=85旧+10新。新增AST检查逐函数核对14个提取函数的定义、签名和完整正文，记录在 [S2_api_map.csv](S2_api_map.csv)。Julia已有断言共158；每批smoke是同一固定trial的逐步回归，不能视为新增独立实验样本。

Golden重放结果保存在：

- [A](S2_work/A_retry2/golden/S1_python_compatibility.json)
- [B](S2_work/B/golden/S1_python_compatibility.json)
- [C](S2_work/C/golden/S1_python_compatibility.json)

这些文件名沿用被调用的、未修改的S1审计函数，但实际位于新的S2输出目录；原S1 JSON/日志没有被覆盖。

## 明确覆盖的契约

- 原生get_θd的形状、数值、递归顺序和d；跨语言CSV/TOML精度、单行/单列、索引/哈希/陈旧输出拒绝。
- 训练均值/RMS及常量列；只改变验证/测试列不能影响拟合均值/尺度；轨迹跨split被拒绝。
- 原尺度系数和偏置恢复、预测等价、Adaptive加权列目标等价与KKT。
- gamma=0数学退化的固定、BIC及验证路径：单位权重、系数、alpha、网格和选中索引一致；公共Adaptive入口仍拒绝gamma=0，与S1一致。
- 单目标结构、重复源、目标在源内、同一边取最大绝对值而非累加、严格边界1e−8与真零；原生解码1–4阶不被误过滤。
- M0原尺度、M0s公共标准化控制；M1/M1v/M2/M3身份不变。
- 零噪声M1/M2 NA理由精确一致，没有fallback或伪系数；选中参数/路径、初始化/权重和候选警告数量匹配。

## 失败记录完整保留

1. [S2_A_python_tests.log](S2_A_python_tests.log)：91项中新增gamma=0跨路径MSE逐位断言失败；85项原有测试通过。先回退2个原文件后，S1快照确认同样8.67e−18差异；只修正错误比较对象。
2. [S2_A_retry1_trial_smoke.log](S2_A_retry1_trial_smoke.log)：91项Python及批量Golden已通过，桥接对直接批量的新增逐位断言失败。再次回退后，S1原桥接复现同一浮点差异；当前与原桥接逐位一致。

详见 [S2_report.md](S2_report.md) 的两次回退与诊断。没有更改算法、提高容差或覆盖失败日志。最终使用相同路径的S1冻结实现做精确比较；跨路径保留原S1容差及支持集约束，所有最终检查通过。

Python新增数据加载测试会显示原load_dataset既有的文件句柄ResourceWarning；函数按原AST提取，没有在重构中顺带修复。该提示保留在日志，与原求解器候选警告分开。原有mock失败测试中打印的 `status=FAILED` 是预期异常测试，不是测试失败。

## 执行边界

没有运行任何会写回历史results的旧audit/logger、没有重跑30-seed、没有修改正式选择锁/指标/警告/图表，没有删除文件。旧manifest包含的历史源码hash不会被新源码冒充；差异严格限于获准6个文件并记录在S2审计中。

本轮新生产/测试/审计文件的哈希索引为 `S2_artifact_manifest.csv`，排除自身；它不替代或覆盖S1和R5历史manifest。

S2结束，停止等待人工验收；S3–S6未执行。
