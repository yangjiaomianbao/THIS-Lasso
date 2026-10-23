# 三体结构恢复指标核验

生成日期：2026-10-23（Asia/Shanghai）。输入目录：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022`。

仅新增本目录派生文件；未重新仿真、拟合、选择阈值或修改实验参数。
输入 trials.csv、summary.csv、edge_scores.csv、REPORT.md 均经实际目录定位；所有计算只使用 order=triple。

## 指标与统计规则

- F1 = 2TP/(2TP+FP+FN)，优先保留 trials.csv 数值，并独立复算。
- AP = Σ ΔRecall × Precision；使用原始 AUPRC_AP，另以 sklearn.average_precision_score 及按同分组累计的独立实现核验全部1365个连续得分。未使用二值分数或梯形PR-AUC。
- MCC = (TP×TN−FP×FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))；分母为0保持NA，原始 MCC_status（包括模型NA时空白）逐种子原样保留。
- Jaccard = TP/(TP+FP+FN)，先逐种子计算再求均值；未从平均F1转换。
- F1/MCC/Jaccard 使用相同预测结构：原始连续得分严格大于原始阈值1e-8；从得分重新生成的混淆矩阵与原文件逐项一致。
- 仅 status=OK 且指标有效的种子纳入该指标汇总。每项保存均值、样本标准差（ddof=1）、有效数；n<2时SD为NA，n=0时均值为NA。CSV空字段表示派生数值NA，原始MCC的nan字符串保存在逐种子表。
- 主表四位小数；CSV和Excel完整精度工作表不提前舍入（保留原始Python双精度数值）。
- 同动力学、同噪声六方法内按未舍入均值比较最佳，绝对容差1e-12视为并列；Excel红色加粗，Markdown加粗，LaTeX红色加粗。NA不参与。
- Markdown、LaTeX和Excel均附完整60行均值±样本标准差版本；Excel另附60行完整精度/状态表、120行逐种子明细和说明。

## 核验结果

- PASS：5动力学 × 2噪声 × 6固定方法 = 60配置，每配置恰好2个不同种子，无重复或遗漏。
- PASS：120个模型中96 OK、24模型级NA、0 FAILED；48配置有有效结果，12配置全部NA。模型NA未当失败或零分。
- PASS：96个有效模型均恰有1365个唯一的有向目标/无序源三体候选及15个真实正例；共核验131040条三体得分，候选集合完整、真值跨方法/噪声一致。
- PASS：F1/AP/Jaccard各96个有效种子值；MCC仅72个有效值，另24个OK模型因分母为0保持MCC_status=NA。模型级NA与指标级NA分别记录。
- PASS：逐种子F1、MCC、AP复算及Jaccard恒等式核验；原summary.csv中的F1/AP均值与新汇总一致。最大误差见validation.json。
- PASS：Excel ZIP/XML读回校验通过，两张展示表各60行；每个红色粗体单元格逐一对照分组最佳值核验。
- PASS：原artifact_manifest.csv的3214项文件大小/SHA256全部通过；生成前后全部3216个现有实验文件的SHA256完全一致。原清单未更新。

## 输出文件

- [threebody_results.csv](threebody_results.csv)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/threebody_results.csv`
- [table1_threebody.xlsx](table1_threebody.xlsx)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/table1_threebody.xlsx`
- [table1_threebody.md](table1_threebody.md)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/table1_threebody.md`
- [table1_threebody.tex](table1_threebody.tex)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/table1_threebody.tex`
- [metrics_validation.md](metrics_validation.md)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/metrics_validation.md`
- [threebody_seed_metrics.csv](threebody_seed_metrics.csv)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/threebody_seed_metrics.csv`
- [validation.json](validation.json)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/validation.json`
- [build_threebody_tables.py](build_threebody_tables.py)：`/home/yg/THIS-Lasso/results/dynamics15/five_systems_certified_20261022/tables/build_threebody_tables.py`

LaTeX为独立可编译的longtable/booktabs文档（含均值及均值±SD两表）；本轮检查源码结构，未调用TeX编译器。
重现仅需以项目Python环境运行本目录build_threebody_tables.py；脚本只读取已有实验、写入本tables目录。

## 输入SHA256

| 文件 | SHA256 |
| --- | --- |
| trials.csv | `9a4870e90a2f46d2c032b46f3687f51d6584b0b05a3a9f053d8dd1f4e629f282` |
| summary.csv | `d0ec96ebea614affac4f2fa75402f21f14c9016746076bc25f79e646b8444691` |
| edge_scores.csv | `50074fcb893ce84e1216b18b8836aba8f0240d6b86a6de3bc8934c4f3b3a5bef` |
| REPORT.md | `05efed0caaebe9d1498640e71c4bc1751c427c63fc624e79a951807217d8adb0` |
| artifact_manifest.csv | `e05c1d63d37f43bfb2b261787b807053154cdefb16d16dc725eabc4961a1298b` |
