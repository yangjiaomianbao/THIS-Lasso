# R0 验收日志

R0 自动执行工作已完成，等待人工验收。最终状态见 [R0 报告](../R0_report.md)。

| 文件 | 结果/用途 |
| --- | --- |
| [julia_smoke.log](julia_smoke.log) | 43 项断言通过，原始 THIS 基本调用成功，退出 0 |
| [python_smoke.log](python_smoke.log) | LassoLarsIC 回归、原始 Su 导入与单节点基本调用成功，退出 0 |
| [python_dependency_check.log](python_dependency_check.log) | pip check 无损坏依赖，退出 0 |
| [reproducibility_check.log](reproducibility_check.log) | 全部 28 个 Python 锁定版本与安装一致，移除 pip 构建机地址，退出 0 |
| [cleanup.log](cleanup.log) | 重复源码核验、历史合并和安装临时文件清理，退出 0 |
| [setup_history.log](setup_history.log) | 25 份历史文件合并，保留原文件名、SHA-256、原始命令/输出/退出码 |

setup_history.log 是多次执行的历史集合，含失败和中断，不应将其中的旧错误视为当前状态。可检索以下段名：

- `python_dependencies_resume.log`：最终 Python 安装成功，退出 0。
- `julia_verify.log`：安装包通过官方 SHA-256 校验。
- `julia_dependencies.log`：Julia 依赖安装及预编译成功，退出 0。
- `environment_lock.log`：生成 Python/Conda 环境锁定文件。
- `activation_check.log`：激活后 Python、Julia 版本与可执行文件路径正确，退出 0。
- `upstream_provenance.log`：固定 HEAD、许可证和逐文件哈希。
- `vscode_install_julia.log`：Julia 扩展 1.249.2 安装成功。

历史中的旧路径、缺失软件结论和构建机地址反映执行时状态，未改写成后来的结果。阶段完成后仅保留合并历史与关键验收记录，避免零散文件持续累积。
