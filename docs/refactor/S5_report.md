# S5：已验证备份的精确路径缓存清理

日期2026-10-20。授权来源为用户“继续执行S3-S6，不需要验收了，给我全部处理完”。本轮按该明确委托完成低风险候选选择和执行，不伪称用户曾逐项阅读清单。授权解释和完整原文见 [S5_authorization.json](S5_authorization.json)。

## 执行结果

删除6,385个单链接Python字节码缓存，合计144,095,231字节（137.42 MiB）；0个源码、配置、报告、科研数据或结果删除。1,140个共享/硬链接Python缓存与44个Julia原生缓存保留。没有通配、递归目录清空或git clean；每个unlink只处理 [批准清单](S5_delete_allowlist_APPROVED.csv) 中已再次核对路径、SHA256、源SHA、mtime和单链接状态的文件。

删除前先建立 [恢复备份](../../archive/caches/S5_python_cache_backup.tar.gz)，6,385个成员全部逐项读回验证SHA256，才写入最终批准记录并执行。备份压缩体积48,832,821字节（46.57 MiB）。[删除日志](S5_deletion_log.csv)逐文件记录授权、日期、原哈希、理由、备份成员与验证结果关联。

缓存逻辑体积减恢复备份，净减少95,262,410字节（90.85 MiB）；按单链接文件原分配块减备份分配块估算减少109,641,728字节（104.56 MiB）。这是缓存清理减备份的口径；全项目净变化还需计入S3–S6全部新增审计和文档，最终另行记录，不能把137.42 MiB说成全项目实际净释放。

## 删除后验证与恢复

15项R3真实桥接测试、3个主要CLI help、全部备份成员再次校验/恢复dry-run、文档链接检查均通过，见 [S5_validation.json](S5_validation.json) 和 [S5_work](S5_work/)。[执行汇总](S5_cleanup_summary.json)中的tests_pending是删除当时的历史字段，完成后的测试状态由validation文件给出，不回写旧记录。

默认恢复检查只读取备份：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-r0/bin/python -B docs/refactor/S5_restore_caches.py
```

如日后确需恢复，另加 `--restore`；该工具只恢复清单中的缺失原缓存，验证归档及每个成员哈希，拒绝覆盖不同内容。正常解释器也可从未变的.py源码重新生成可选缓存。本轮只运行dry-run，未把缓存重新写回。

原科研证据和已有审计文档均保留。S5完成，按用户授权继续S6最终复现审计，无需中间验收。
