"""Read-only metric audit; writes derived tables only beside this script.

Run with the project's .venv-r0/bin/python -B. No fitting imports are used.
XLSX is written with standard-library OOXML to avoid changing the environment.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations, product
import csv
import hashlib
import json
import math
import statistics
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import numpy as np
from sklearn.metrics import average_precision_score

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SYSTEMS = {'kuramoto': 'Kuramoto', 'lorenz': 'Lorenz',
           'lotka_volterra': 'Lotka–Volterra', 'stuart_landau': 'Stuart–Landau', 'sis': 'SIS'}
METHODS = {'M0D': 'STLSQ raw', 'M0sD': 'STLSQ std', 'M1B': 'LASSO BIC',
           'M2B': 'Adaptive BIC', 'M1vC': 'LASSO val', 'M3C': 'Adaptive val'}
METRICS = ['F1', 'AP', 'MCC', 'Jaccard']
TITLE = 'Table 1. Performance Comparison for Three-body Interaction Reconstruction'
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'


def read(name):
    with (ROOT / name).open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return row['system'], float(row['noise_ratio']), row['method'], int(row['seed'])


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def numeric(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def manual_ap(truth, scores):
    # Group tied scores; use precision at the end of each tie group.
    order = np.argsort(-scores, kind='stable')
    y, s = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.cumsum(y)[ends]
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0., recall]) * tp / (ends+1)))


def col(i):
    result = ''
    while i:
        i, remainder = divmod(i-1, 26)
        result = chr(65+remainder) + result
    return result


def write_xlsx(sheets):
    """Each sheet: name, list of rows, {(row,column): style_id}."""
    styles = f'''<styleSheet xmlns="{NS}">
    <numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts>
    <fonts count="4"><font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><color rgb="FFFF0000"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
    <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF24476A"/><bgColor indexed="64"/></patternFill></fill></fills>
    <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
    <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
    <cellXfs count="6">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="164" fontId="2" fillId="0" borderId="0" xfId="0" applyNumberFormat="1" applyFont="1"/>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    </cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    relns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    offrel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    parts = {'xl/styles.xml': styles,
             '_rels/.rels': f'<Relationships xmlns="{relns}"><Relationship Id="rId1" Type="{offrel}/officeDocument" Target="xl/workbook.xml"/></Relationships>'}
    wb, rels, content = [], [], []
    for i, (name, rows, marks) in enumerate(sheets, 1):
        wb.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        rels.append(f'<Relationship Id="rId{i}" Type="{offrel}/worksheet" Target="worksheets/sheet{i}.xml"/>')
        content.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        xmlrows = []
        for r, values in enumerate(rows, 1):
            cells = []
            for c, value in enumerate(values, 1):
                style = marks.get((r,c), 1 if r == 1 else 0)
                attrs = f'r="{col(c)}{r}" s="{style}"'
                if isinstance(value, (int, float)):
                    assert math.isfinite(value)
                    cells.append(f'<c {attrs}><v>{value!r}</v></c>')
                else:
                    cells.append(f'<c {attrs} t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>')
            xmlrows.append(f'<row r="{r}">' + ''.join(cells) + '</row>')
        width = len(rows[0])
        extra_cols = f'<col min="3" max="{width}" width="24" customWidth="1"/>' if width >= 3 else ''
        parts[f'xl/worksheets/sheet{i}.xml'] = f'''<worksheet xmlns="{NS}"><dimension ref="A1:{col(width)}{len(rows)}"/>
        <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
        <cols><col min="1" max="1" width="23" customWidth="1"/><col min="2" max="2" width="12" customWidth="1"/>{extra_cols}</cols>
        <sheetData>{''.join(xmlrows)}</sheetData><autoFilter ref="A1:{col(width)}{len(rows)}"/>
        <pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>
        <pageSetup paperSize="9" orientation="landscape" fitToWidth="1" fitToHeight="0"/></worksheet>'''
    rels.append(f'<Relationship Id="rId{len(sheets)+1}" Type="{offrel}/styles" Target="styles.xml"/>')
    parts['xl/workbook.xml'] = f'<workbook xmlns="{NS}" xmlns:r="{offrel}"><sheets>{"".join(wb)}</sheets></workbook>'
    parts['xl/_rels/workbook.xml.rels'] = f'<Relationships xmlns="{relns}">{"".join(rels)}</Relationships>'
    parts['[Content_Types].xml'] = '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(content) + '</Types>'
    with zipfile.ZipFile(OUT / 'table1_threebody.xlsx', 'w', zipfile.ZIP_DEFLATED) as z:
        for name, text in parts.items():
            ET.fromstring(text)
            z.writestr(name, '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + text)


def main():
    # Baseline hashes cover every pre-existing file in this batch, including caches.
    original = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*')
                if p.is_file() and OUT not in p.parents}
    manifest = read('artifact_manifest.csv')
    for item in manifest:
        assert original[item['path']] == item['sha256'], item['path']
        assert (ROOT/item['path']).stat().st_size == int(item['size'])
    required = {'system','seed','noise_ratio','method','status','order','TP','FP','FN','TN',
                'F1','MCC','MCC_status','AUPRC_AP','AUPRC_status','threshold'}
    trials = read('trials.csv')
    assert required <= trials[0].keys()
    trials = [r for r in trials if r['order'] == 'triple']
    seeds = sorted({int(r['seed']) for r in trials})
    assert len(seeds) == 2
    expected = list(product(SYSTEMS, [0., .05], METHODS))
    lookup = {key(r): r for r in trials}
    assert len(lookup) == len(trials) == 120
    assert set(lookup) == {(*g, seed) for g in expected for seed in seeds}
    assert Counter(r['status'] for r in trials) == {'OK':96, 'NA':24}
    edge_groups = defaultdict(list)
    for r in read('edge_scores.csv'):
        if r['order'] == 'triple':
            edge_groups[key(r)].append(r)
    assert set(edge_groups) == {k for k,r in lookup.items() if r['status']=='OK'}
    candidates = {(i,j,k) for i in range(1,16)
                  for j,k in combinations([x for x in range(1,16) if x!=i],2)}
    assert len(candidates) == 1365
    seed_rows, errors, truths = [], defaultdict(list), {}
    for g in expected:
        for seed in seeds:
            r = lookup[(*g,seed)]
            a = dict(r)
            a.update(method_name=METHODS[r['method']], AP=r['AUPRC_AP'], Jaccard='',
                     Jaccard_status='', MCC_diagnostic='', validation='MODEL_NA_PRESERVED')
            status_path = Path(r['system'])/f'seed_{seed}'/f'noise_{round(g[1]*1000):03d}'/r['method']/'status.json'
            status = json.loads((ROOT/status_path).read_text())
            assert status['status'] == r['status']
            a['model_status_path'] = str(status_path)
            a['model_node_counts'] = json.dumps(status.get('node_counts'), sort_keys=True)
            a['model_status_json'] = json.dumps(status, ensure_ascii=False, sort_keys=True)
            if r['status'] == 'OK':
                edges = edge_groups[(*g,seed)]
                ids = [(int(x['target']), *sorted([int(x['source_1']),int(x['source_2'])])) for x in edges]
                assert len(ids)==1365 and len(set(ids))==1365 and set(ids)==candidates
                assert all(x['status']=='OK' and x['truth'] in ('0','1') for x in edges)
                y = np.array([int(x['truth']) for x in edges])
                scores = np.array([float(x['score']) for x in edges])
                assert np.isfinite(scores).all() and y.sum()==15
                positive = {i for i,t in zip(ids,y) if t}
                tkey = (g[0],seed)
                assert truths.setdefault(tkey,positive)==positive
                threshold = float(r['threshold'])
                assert threshold==1e-8
                pred = scores > threshold
                counts = [int(np.sum(pred & (y==1))),int(np.sum(pred & (y==0))),
                          int(np.sum(~pred & (y==1))),int(np.sum(~pred & (y==0)))]
                assert counts == [int(r[x]) for x in ['TP','FP','FN','TN']]
                tp,fp,fn,tn = counts
                assert sum(counts)==int(r['candidate_count'])==1365
                assert tp+fn==int(r['positive_count'])==15
                f1 = 2*tp/(2*tp+fp+fn)
                near(f1,r['F1']); errors['F1'].append(abs(f1-float(r['F1'])))
                ap = average_precision_score(y,scores)
                near(ap,r['AUPRC_AP']); near(manual_ap(y,scores),ap)
                assert r['AUPRC_status']=='OK'
                errors['AP'].append(abs(ap-float(r['AUPRC_AP'])))
                denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
                if denom:
                    mcc = (tp*tn-fp*fn)/denom
                    assert r['MCC_status']=='OK'
                    near(mcc,r['MCC']); errors['MCC'].append(abs(mcc-float(r['MCC'])))
                else:
                    assert r['MCC_status']=='NA' and numeric(r['MCC']) is None
                    a['MCC_diagnostic']='Undefined: denominator is zero; original MCC_status=NA retained.'
                a.update(Jaccard=tp/(tp+fp+fn),Jaccard_status='OK', validation='PASS')
                near(a['Jaccard'], f1/(2-f1))  # Seed-wise identity only.
            else:
                assert all(numeric(r[x]) is None for x in ['TP','FP','FN','TN','F1','MCC','AUPRC_AP'])
            seed_rows.append(a)
    aggregates=[]
    old_summary = {(r['system'],float(r['noise_ratio']),r['method']):r
                   for r in read('summary.csv') if r['order']=='triple'}
    for g in expected:
        rows=[r for r in seed_rows if key(r)[:3]==g]
        states=Counter(r['status'] for r in rows)
        result=dict(system=g[0],Dynamics=SYSTEMS[g[0]],noise_ratio=g[1],Noise=f'{g[1]:.0%}',
                    method=g[2],Method=METHODS[g[2]],order='triple',attempted=2,
                    OK=states['OK'],NA=states['NA'],FAILED=states['FAILED'],
                    status='OK' if states['OK']==2 else 'NA' if states['NA']==2 else 'PARTIAL',
                    seed_statuses=json.dumps({r['seed']:r['status'] for r in rows},sort_keys=True))
        for metric in METRICS:
            vals=[numeric(r[metric]) for r in rows if r['status']=='OK' and
                  (metric!='MCC' or r['MCC_status']=='OK') and
                  (metric!='AP' or r['AUPRC_status']=='OK')]
            vals=[v for v in vals if v is not None]
            result.update({metric+'_mean':statistics.mean(vals) if vals else '',
                           metric+'_std':statistics.stdev(vals) if len(vals)>1 else '',
                           metric+'_n':len(vals)})
        for field in ['MCC_status','AUPRC_status','MCC_diagnostic','AUPRC_reason','model_status_path']:
            result[field+'_by_seed']=json.dumps({r['seed']:r[field] for r in rows},ensure_ascii=False,sort_keys=True)
        prev=old_summary[g]
        for field in ['OK','NA','FAILED']:
            assert result[field]==int(prev[field])
        if states['OK']:
            near(result['F1_mean'],prev['F1_mean']);near(result['AP_mean'],prev['AUPRC_AP_mean'])
        aggregates.append(result)
    assert len(aggregates)==60 and sum(r['NA'] for r in aggregates)==24
    assert sum(r['MCC_n'] for r in aggregates)==72
    write_csv('threebody_results.csv',aggregates)
    write_csv('threebody_seed_metrics.csv',seed_rows)
    winners=set()
    for system,noise in product(SYSTEMS,[0.,.05]):
        block=[(i,r) for i,r in enumerate(aggregates) if (r['system'],r['noise_ratio'])==(system,noise)]
        for m in METRICS:
            values=[r[m+'_mean'] for _,r in block if r[m+'_n']]
            if values:
                best=max(values)
                winners.update((i,m) for i,r in block if r[m+'_n'] and
                               math.isclose(r[m+'_mean'],best,rel_tol=0,abs_tol=1e-12))

    def display(r,m,sd=False):
        n=r[m+'_n']
        value=f'{r[m+"_mean"]:.4f}' if n else 'NA'
        if sd and n:
            value += ' ± ' + (f'{r[m+"_std"]:.4f}' if n>1 else 'NA')
        if n<2:
            value += f' [n={n}/2]'
        return value

    headers=['Dynamics','Noise','Method','F1 ↑','AP ↑','MCC ↑','Jaccard ↑','OK/2']
    md=[f'# {TITLE}','',
        '仅评价三体结构；数值为有效种子均值。NA 不计为 0。均值±标准差使用样本标准差（ddof=1）。',
        '最优值按完整精度在同动力学、同噪声的六方法内逐指标比较（绝对并列容差 1e-12），以粗体标记。',
        '`[n=k/2]` 为该指标有效种子数；`OK/2` 为模型有效种子数。各指标计数与原始状态见 CSV 和 Excel 状态表。','']
    tex=[r'\documentclass{article}',r'\usepackage[a4paper,landscape,margin=1.2cm]{geometry}',
         r'\usepackage{booktabs,longtable,xcolor}',r'\begin{document}',r'\small']
    sheets=[]
    for sd in [False,True]:
        md += ['## '+('Mean ± sample SD' if sd else 'Mean'),'',
               '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*3+['---:']*5)+' |']
        tex += [r'\begin{longtable}{lllrrrrr}',
                r'\caption{'+TITLE+(' (mean $\pm$ sample SD)' if sd else ' (mean)')+r'}\\',
                r'\toprule',r'Dynamics & Noise & Method & F1 $\uparrow$ & AP $\uparrow$ & MCC $\uparrow$ & Jaccard $\uparrow$ & OK/2 \\',
                r'\midrule',r'\endfirsthead',r'\toprule',
                r'Dynamics & Noise & Method & F1 $\uparrow$ & AP $\uparrow$ & MCC $\uparrow$ & Jaccard $\uparrow$ & OK/2 \\',
                r'\midrule',r'\endhead',r'\bottomrule',r'\endfoot']
        xrows=[headers]; marks={}
        for i,r in enumerate(aggregates):
            labels=[r['Dynamics'],r['Noise'],r['Method']]
            values=[display(r,m,sd) for m in METRICS]
            mdvalues=[f'**{v}**' if (i,m) in winners else v for m,v in zip(METRICS,values)]
            md.append('| '+' | '.join(labels+mdvalues+[f'{r["OK"]}/2'])+' |')
            texvalues=[v.replace(' ± ',r' $\pm$ ') for v in values]
            texvalues=[r'\textcolor{red}{\textbf{'+v+'}}' if (i,m) in winners else v for m,v in zip(METRICS,texvalues)]
            texlabels=[v.replace('–','--').replace('%',r'\%') for v in labels]
            if i and i%6==0:tex.append(r'\midrule')
            tex.append(' & '.join(texlabels+texvalues+[f'{r["OK"]}/2'])+r' \\')
            xv=[]
            for j,m in enumerate(METRICS,4):
                value=display(r,m,sd) if sd or r[m+'_n']!=2 else r[m+'_mean']
                xv.append(value)
                marks[(i+2,j)] = (3 if isinstance(value,float) else 4) if (i,m) in winners else (2 if isinstance(value,float) else 0)
            xrows.append(labels+xv+[f'{r["OK"]}/2'])
        md.append('')
        tex += [r'\end{longtable}',r'\noindent NA is excluded from means; [n=k/2] denotes the metric-specific valid seed count. ',
                r'SD uses ddof=1. Red bold indicates the best mean within each dynamics/noise group, including ties (absolute tolerance $10^{-12}$).',r'\clearpage']
        sheets.append(('Table1 Means' if not sd else 'Table1 Mean SD',xrows,marks))
    tex.append(r'\end{document}')
    (OUT/'table1_threebody.md').write_text('\n'.join(md)+'\n')
    (OUT/'table1_threebody.tex').write_text('\n'.join(tex)+'\n')
    for name, rows in [('Status and Full Precision',aggregates),('Seed Metrics',seed_rows)]:
        sheets.append((name,[list(rows[0])]+[list(r.values()) for r in rows],{}))
    notes=[['Item','Definition'],['Scope','15 nodes; 1365 directed target/unordered-source triple candidates; 15 positives'],
           ['Threshold','Original strict score > 1e-8; unchanged; no label-based threshold selection'],
           ['F1','2TP/(2TP+FP+FN)'],['AP','Original AUPRC_AP, checked with all continuous scores; not trapezoidal PR-AUC'],
           ['MCC','Original MCC and MCC_status preserved; denominator zero -> NA'],
           ['Jaccard','TP/(TP+FP+FN), computed per seed before aggregation'],
           ['SD','Sample standard deviation, ddof=1; undefined for n<2'],
           ['Best','Within same dynamics/noise, compare unrounded means; ties abs tol=1e-12'],
           ['Missing','24 model-level NA; another 24 OK models have undefined MCC'],
           ['Source',str(ROOT)],['Created','2026-10-23 Asia/Shanghai']]
    sheets.append(('Notes',notes,{}))
    write_xlsx(sheets)
    # Reload XLSX package to validate row counts, exact values and all red/bold marks.
    with zipfile.ZipFile(OUT/'table1_threebody.xlsx') as z:
        assert z.testzip() is None
        for name in z.namelist():ET.fromstring(z.read(name))
        style=ET.fromstring(z.read('xl/styles.xml'))
        fonts=style.find(f'{{{NS}}}fonts')
        assert fonts[2].find(f'{{{NS}}}b') is not None
        assert fonts[2].find(f'{{{NS}}}color').get('rgb')=='FFFF0000'
        for sheet_id in [1,2]:
            sheet=ET.fromstring(z.read(f'xl/worksheets/sheet{sheet_id}.xml'))
            rows=sheet.find(f'{{{NS}}}sheetData')
            assert len(rows)==61
            for i,row in enumerate(list(rows)[1:]):
                for j,m in enumerate(METRICS,3):
                    cell=row[j]
                    assert (int(cell.get('s')) in (3,4))==((i,m) in winners)
                    value=cell.find(f'{{{NS}}}v')
                    if value is not None:near(value.text,aggregates[i][m+'_mean'])
    assert len(list(csv.DictReader((OUT/'threebody_results.csv').open())))==60
    # Validate every existing file again after generation; never update old manifests.
    after={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and OUT not in p.parents}
    assert original==after, 'Original experiment files changed'
    audit=dict(status='PASS',configurations=60,seed_models=120,OK=96,model_NA=24,FAILED=0,
               configurations_with_OK=48,configurations_all_NA=12,
               metric_valid_seed_counts={m:sum(r[m+'_n'] for r in aggregates) for m in METRICS},
               undefined_MCC_in_OK=24,original_files_unchanged=len(original),
               original_manifest_entries_verified=len(manifest),
               maximum_metric_recompute_difference={m:max(v) for m,v in errors.items()},
               candidate_score_rows_checked=96*1365,highlighted_cells_per_table=len(winners),
               input_sha256={n:original[n] for n in ['trials.csv','summary.csv','edge_scores.csv','REPORT.md','artifact_manifest.csv']})
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    report=['# 三体结构恢复指标核验','',f'生成日期：2026-10-23（Asia/Shanghai）。输入目录：`{ROOT}`。','',
            '仅新增本目录派生文件；未重新仿真、拟合、选择阈值或修改实验参数。',
            '输入 trials.csv、summary.csv、edge_scores.csv、REPORT.md 均经实际目录定位；所有计算只使用 order=triple。','',
            '## 指标与统计规则','',
            '- F1 = 2TP/(2TP+FP+FN)，优先保留 trials.csv 数值，并独立复算。',
            '- AP = Σ ΔRecall × Precision；使用原始 AUPRC_AP，另以 sklearn.average_precision_score 及按同分组累计的独立实现核验全部1365个连续得分。未使用二值分数或梯形PR-AUC。',
            '- MCC = (TP×TN−FP×FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))；分母为0保持NA，原始 MCC_status（包括模型NA时空白）逐种子原样保留。',
            '- Jaccard = TP/(TP+FP+FN)，先逐种子计算再求均值；未从平均F1转换。',
            '- F1/MCC/Jaccard 使用相同预测结构：原始连续得分严格大于原始阈值1e-8；从得分重新生成的混淆矩阵与原文件逐项一致。',
            '- 仅 status=OK 且指标有效的种子纳入该指标汇总。每项保存均值、样本标准差（ddof=1）、有效数；n<2时SD为NA，n=0时均值为NA。CSV空字段表示派生数值NA，原始MCC的nan字符串保存在逐种子表。',
            '- 主表四位小数；CSV和Excel完整精度工作表不提前舍入（保留原始Python双精度数值）。',
            '- 同动力学、同噪声六方法内按未舍入均值比较最佳，绝对容差1e-12视为并列；Excel红色加粗，Markdown加粗，LaTeX红色加粗。NA不参与。',
            '- Markdown、LaTeX和Excel均附完整60行均值±样本标准差版本；Excel另附60行完整精度/状态表、120行逐种子明细和说明。','',
            '## 核验结果','',
            '- PASS：5动力学 × 2噪声 × 6固定方法 = 60配置，每配置恰好2个不同种子，无重复或遗漏。',
            '- PASS：120个模型中96 OK、24模型级NA、0 FAILED；48配置有有效结果，12配置全部NA。模型NA未当失败或零分。',
            '- PASS：96个有效模型均恰有1365个唯一的有向目标/无序源三体候选及15个真实正例；共核验131040条三体得分，候选集合完整、真值跨方法/噪声一致。',
            '- PASS：F1/AP/Jaccard各96个有效种子值；MCC仅72个有效值，另24个OK模型因分母为0保持MCC_status=NA。模型级NA与指标级NA分别记录。',
            '- PASS：逐种子F1、MCC、AP复算及Jaccard恒等式核验；原summary.csv中的F1/AP均值与新汇总一致。最大误差见validation.json。',
            '- PASS：Excel ZIP/XML读回校验通过，两张展示表各60行；每个红色粗体单元格逐一对照分组最佳值核验。',
            f'- PASS：原artifact_manifest.csv的{len(manifest)}项文件大小/SHA256全部通过；生成前后全部{len(original)}个现有实验文件的SHA256完全一致。原清单未更新。','',
            '## 输出文件','']
    for name in ['threebody_results.csv','table1_threebody.xlsx','table1_threebody.md','table1_threebody.tex',
                 'metrics_validation.md','threebody_seed_metrics.csv','validation.json','build_threebody_tables.py']:
        report.append(f'- [{name}]({name})：`{OUT/name}`')
    report += ['', 'LaTeX为独立可编译的longtable/booktabs文档（含均值及均值±SD两表）；本轮检查源码结构，未调用TeX编译器。',
               '重现仅需以项目Python环境运行本目录build_threebody_tables.py；脚本只读取已有实验、写入本tables目录。','',
               '## 输入SHA256','', '| 文件 | SHA256 |','| --- | --- |']
    report += [f'| {n} | `{h}` |' for n,h in audit['input_sha256'].items()]
    (OUT/'metrics_validation.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
