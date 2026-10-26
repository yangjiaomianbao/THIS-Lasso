"""Estimate proposed work from saved runtimes; does not generate data or fit models."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/yg/THIS-Lasso')
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'results/dynamics15'


def main():
    fair = BASE / 'fair_comparison_v2_full_run01'
    old = BASE / 'five_systems_certified_20261022'
    with (fair / 'resources.csv').open() as f:
        rows = list(csv.DictReader(f))
    completion = json.loads((fair / 'completion.json').read_text())
    cpu_h = sum(float(r['cpu_seconds']) for r in rows) / 3600
    wall_h = completion['wall_seconds'] / 3600
    throughput = cpu_h / wall_h
    val_zero = sum(float(r['cpu_seconds']) for r in rows
                   if r['method'] != 'STLSQ_val' and float(r['noise_ratio']) == 0) / 3600
    val_five = sum(float(r['cpu_seconds']) for r in rows
                   if r['method'] != 'STLSQ_val' and float(r['noise_ratio']) == .05) / 3600
    old_total = json.loads((old / 'completion.json').read_text())
    old_node_h = {}
    for f in old.glob('*/seed_*/noise_*/M*/node_*.json'):
        a = json.loads(f.read_text())
        key = f'{f.parts[-3]}/{f.parts[-2]}'
        old_node_h[key] = old_node_h.get(key, 0.) + a.get('wall_seconds', 0.) / 3600
    scenarios = []
    for multiplier in [1., 1.5, 2.]:
        cpu_projection = val_zero + val_five * (1 + multiplier)
        scenarios.append(dict(ten_percent_cost_relative_to_five_percent=multiplier,
                              A_validation_cpu_h=cpu_projection,
                              A_validation_wall_h_at_historical_throughput=cpu_projection / throughput,
                              B_validation_wall_h_at_historical_throughput=cpu_projection / throughput * 50 / 30))
    coefficient_cells_per_five_systems = 3 * 15 * 816 + 30 * 5456 + 45 * 17296
    result = dict(
        estimate_date='2026-10-26 Asia/Shanghai',
        scope='Review and runtime estimation only; no new fitting or data generation.',
        evidence=dict(old_models=old_total['models'], old_wall_h=old_total['wall_seconds'] / 3600,
                      fair_models=completion['models'], fair_wall_h=wall_h,
                      fair_equation_cpu_h=cpu_h, fair_effective_cpu_throughput=throughput,
                      fair_validation_zero_cpu_h=val_zero, fair_validation_five_cpu_h=val_five,
                      old_node_wall_h_by_noise_method=old_node_h),
        assumptions=dict(workers=8, blas_threads=1, same_hardware_and_solver=True,
                         new_ten_percent_noise_is_not_yet_benchmarked=True,
                         fitting_paths_exported_during_new_fits=True,
                         no_five_fold_CV_or_full_group_deletion_search=True),
        sizes=dict(A_model_records=2700, A_expected_BIC_NA=480, A_potentially_usable_models_before_failures=2220,
                   old_records_reusable_if_hash_and_semantics_match=120,
                   old_OK_final_coefficients_reusable=96, A_new_configuration_records=2580,
                   A_validation_equation_paths=21600, A_validation_grid_points=259200,
                   B_additional_model_records=4500, total_model_records=7200,
                   A_dense_validation_grid_coefficients_GiB=coefficient_cells_per_five_systems * 30 * 3 * 2 * 12 * 8 / 2**30),
        validation_only_scenarios=scenarios,
        budget_wall_hours=dict(existing_support_audit=[.25, .75], A_incremental_data_and_fits=[18, 36],
                               A_path_decode_statistics=[3, 8], conditional_bounded_Q4=[2, 6],
                               B_additional_data_and_fits=[30, 60], B_path_statistics=[5, 12]),
        limitations=['Budget ranges are planning estimates, not statistical confidence intervals.',
                     'New code implementation and validation effort is excluded.',
                     '10% noise, tails, IO, source compatibility and Q4 refitting remain unbenchmarked.',
                     'Per-node wall-time sums overlap under concurrency and are not batch wall time.',
                     'Q4 has no finite estimate unless group count, competitor size and refit policy are bounded.'])
    result['evidence_files_sha256'] = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [fair / 'resources.csv', fair / 'completion.json', old / 'completion.json']}
    (OUT / 'runtime_estimate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k in ('evidence', 'sizes', 'validation_only_scenarios', 'budget_wall_hours')}, indent=2))


if __name__ == '__main__':
    main()
