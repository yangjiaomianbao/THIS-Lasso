# S3：文档导航与原位归档

日期2026-10-20。用户已明确授权连续执行S3–S6且无需中间验收，本轮不在阶段边界重复询问。

- 新增 [docs/README.md](../README.md)，唯一当前科研主报告仍为 [R5_report.md](../../reports/R5_report.md)，未改其结论、数字或原始链接。
- 刷新根README，消除“尚未启动R3”的过期状态；完整旧字节保存在 [README_before_S3.md](../../archive/snapshots/README_before_S3.md)。
- 新建 [archive/INDEX.md](../../archive/INDEX.md) 与 [1800项逐文件归档映射](../../archive/S3_archive_map.csv)：历史报告、环境记录、smoke、中止批次和正式history保持原位；0个文件移动、0个文件删除。
- 显式保留M0s/M1v，R4 B与E2/E3仍标为未运行、需独立科研授权。
- [链接检查](S3_link_report.csv) 覆盖346处本地链接：273直接可达、73处历史相对链接经来源映射/说明可达，未解决项0。包括56处原历史问题和旧README复制后的17处相对路径。

原history快照和合并安装日志不回写；[archive/LINKS.md](../../archive/LINKS.md) 给出实际可达路径。两处退休环境材料明确标明合并记录/功能替代，不冒充原字节副本。历史原文中的相对链接仍保留原写法，使用伴随导航访问。

原论文目录、算法、配置、所有科研报告及实验结果未变；本阶段唯一原文件改动是已备份的根README。预检见 [S3_preflight.json](S3_preflight.json)。S3完成，按用户授权继续S4，不要求中间验收。
