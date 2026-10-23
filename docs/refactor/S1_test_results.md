# S1本轮测试与兼容检查

日期：2026-10-19，Asia/Shanghai。本文件记录本轮实际执行结果，不沿用历史日志冒充新测试。

## 已有测试

| 检查 | 实际命令主体 | 结果 | 本轮日志 |
| --- | --- | --- | --- |
| Python R2–R5 | `.venv-r0/bin/python -B -m unittest discover -s tests -p 'test_r*.py' -v` | 85项通过，0失败，退出0 | [S1_python_tests.log](S1_python_tests.log) |
| Julia R1 | `julia --startup-file=no --history-file=no --compiled-modules=no --project=environment/r0/julia tests/test_r1_baseline.jl` | 68项断言通过，退出0 | [S1_julia_r1_tests.log](S1_julia_r1_tests.log) |
| Julia R1 revision | 同参数，`tests/test_r1_revision.jl` | 27项断言通过，退出0 | [S1_julia_r1_revision_tests.log](S1_julia_r1_revision_tests.log) |
| Julia R3 | 同参数，`tests/test_r3_adapter.jl` | 63项断言通过，退出0 | [S1_julia_r3_tests.log](S1_julia_r3_tests.log) |

已有Python分项：R2 Adaptive=30，R3 bridge=15，R4 comparison=17，R5 noise=14，R5 batch=9。已有Julia共158项断言。覆盖目标函数/权重/KKT、guarded BIC、NA、验证拆分、CSV精度、原生解码、标准化还原、边全集/AP和配对统计等。

Python日志中的 `E1_FIT ... status=FAILED` 是原测试有意注入的KKT异常，验证失败记录不能变成伪系数；该测试显示ok，不是实际测试失败。未跳过或改写测试。

## 最小兼容重放

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| R5 seed22261035、0%/5% 的Python M1/M2/M3 | 全通过；0% M1/M2各8节点NA理由精确一致；其余恢复系数差0 | [S1_python_compatibility.log](S1_python_compatibility.log)、[JSON](S1_python_compatibility.json) |
| R4 seed22261017 的 M1v | 通过，系数/选中alpha差0，支持和结构计数一致 | 同上 |
| R5 paired输入的原生theta/d/标准化 | 与保存值完全相等 | [S1_julia_compatibility.log](S1_julia_compatibility.log) |
| R5 paired M0/STLSQ及全部有效模型解码 | 通过，M0系数差0、所有1–4阶结构键和值匹配 | 同上 |
| R4 M0s控制 | 通过，标准化和还原后系数差0，支持/解码一致 | 同上 |
| 新增S1 Julia兼容断言 | 74项通过，退出0 | 同上 |

比较中的alpha、初值、权重、选中路径索引、候选路径及警告数量均保持；Python有效重放系数逐位相等。M0/M0s原生系数最大绝对差也是0。所有拟合只在内存中；没有将新系数或新指标写入正式results。

兼容检查仅固定一个paired R5 seed加一个既有R4控制seed，不是30-seed重跑，不是新增R4 B/E2/E3实验，也未把重放表现用于改参数。

## 冻结与统计哨兵

- 全项目清单31,479条；正式R5清单4,113文件；两上游118个跟踪文件；222成员源码快照逐成员哈希检查通过，见 [S1_freeze.log](S1_freeze.log)。
- Golden111文件，已有应有字段齐全，未补造输出，见 [S1_golden.log](S1_golden.log)。
- 原始trials.csv只读重算均值，12个5%/10%/20%三体F1哨兵均与原summary及任务包四位小数一致，见 [S1_sentinel_checks.json](S1_sentinel_checks.json)。
- 420 OK、60零噪声BIC NA、0 FAILED保持。原候选警告979、选中警告0保持；所有原诊断/警告文件继续由冻结哈希保护，没有重新生成候选警告表。
- 收尾完整性核查通过：31,479个原文件/链接无变化，正式R5全部4,113文件保持；见 [S1_integrity_check.json](S1_integrity_check.json) 与 [S1_integrity.log](S1_integrity.log)，passed=true、errors=[]。

## 执行环境与副作用控制

实际解释器/Julia路径、完整命令、开始/结束时间、隔离环境变量和EXIT_CODE均在各日志中。Python3.11.17、Julia1.9.4，BLAS/Julia单线程；`PYTHONDONTWRITEBYTECODE=1`，Julia禁历史/持久编译模块。测试生成的临时工作目录仅在 `docs/refactor/S1_work/`，已有测试按原有清理机制回收自身临时文件；未清理任何原文件。

未调用 `r3_audit.py`、`r4_audit.py`、`r5_audit.py`、`r5_batch_audit.py` 或会写回正式independence文件的原入口，未使用历史results目录的logger。用于本轮的审计驱动和日志均为S1新增文件。

日志/JSON以独占创建保护，本轮没有重试覆盖失败日志或放宽阈值。未来复查应使用新的获准输出路径/日志标签，不能直接覆盖这套基线。

S1验收范围内未发现测试失败或不可归因差异。S1后停止，等待人工验收与单独批准S2。
