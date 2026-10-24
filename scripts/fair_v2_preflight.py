"""S6 read-only readiness checks; NEVER starts the formal experiment."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import tomllib
from src.experiments.fair_v2 import read_plan,SYSTEMS,METHODS,FRACTIONS,load_csv,sha,save_json
from src.bridge.numeric_io import write_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=ROOT/'configs/fair_comparison_v2')
    p.add_argument('--coverage',type=Path,default=ROOT/'results/dynamics15/fair_v2_five_systems_smoke_20261023_v1')
    p.add_argument('--output',type=Path,default=ROOT/'docs/experiments/fair_comparison_v2/S6_readiness.json')
    a=p.parse_args();c=read_plan(a.plan)
    assert c['systems']==SYSTEMS and c['method_order']==METHODS
    assert [c[k] for k in ['physical_nodes','dmax','train_samples','validation_samples','test_samples']]==[15,3,1024,256,256]
    assert c['noise_ratios']==[0.,.05] and c['decode_threshold']==1e-8
    assert c['lasso']['alpha_fractions']==c['stlsq']['alpha_fractions']==FRACTIONS
    assert c['adaptive']['ridge_rho']==c['stlsq']['rho']==1.
    assert c['adaptive']['gamma']==1. and c['adaptive']['eps']==1e-6
    assert not c['adaptive']['standardize'] and c['adaptive']['fit_intercept']
    assert c['lasso']['kkt_tolerance']==1e-6 and c['lasso']['tol']==1e-10 and c['lasso']['max_iter']==100000
    assert c['stlsq']['niter']==10 and c['stlsq']['validation_tolerance']==c['lasso']['validation_tolerance']==1e-12
    rs=c['repetitions'];assert len(rs)==30 and set(Counter(r['network_id'] for r in rs).values())=={3}
    assert len({r['network_seed'] for r in rs})==10 and len({r['data_seed'] for r in rs})==30
    baseline=tomllib.loads((ROOT/'configs/dynamics15_five_systems_certified_v1_1.toml').read_text())
    for s in SYSTEMS:
        for k,v in c[s].items():assert baseline[s][k]==v,(s,k)
    complete=json.loads((a.coverage/'completion.json').read_text())
    assert complete['mode']=='coverage' and complete['models']==30 and complete['counts']=={'OK':30,'NA':0,'FAILED':0}
    assert not complete['full_experiment_run']
    contract=json.loads((a.coverage/'run_contract.json').read_text())
    assert contract['plan_sha256']==sha(a.plan/'config_locked.toml')
    for name,h in contract['source_sha256'].items():assert sha(ROOT/name)==h,name
    selected=[r for r in load_csv(a.coverage/'selection_details.csv') if r['selected']=='True']
    boundary=[]
    for system in SYSTEMS:
        for ratio in [0.,.05]:
            for method in METHODS:
                rows=[r for r in selected if (r['system'],float(r['noise_ratio']),r['method'])==(system,ratio,method)]
                boundary.append(dict(system=system,noise_ratio=ratio,method=method,target_equations=len(rows),
                    largest_parameter_selected=sum(int(r['candidate'])==0 for r in rows),
                    smallest_parameter_selected=sum(int(r['candidate'])==11 for r in rows),
                    relative_parameter_counts=json.dumps(dict(Counter(r['relative_parameter'] for r in rows)),sort_keys=True)))
    write_rows(a.output.parent/'grid_boundary_diagnostics.csv',boundary)
    result=dict(status='READY_FOR_USER_CONFIRMATION',planned_models=900,network_realizations=10,data_repetitions=30,
        source_and_plan_match_tested_coverage=True,coverage_models=30,coverage_independent_networks=1,
        full_experiment_run=False,coverage_wall_seconds=complete['wall_seconds'],coverage_resources=complete['resources'],
        rough_30x_coverage_hours=complete['wall_seconds']*30/3600,
        timing_limitation='Single network extrapolation only; difficult paths/fallback can substantially increase duration.',
        grid_policy='Record boundary selections; do not retune grid using structure metrics.',
        selection_warning='Validation MSE selects predictive models, not structure-F1-optimal models.',
        config_sha256=sha(a.plan/'config_locked.toml'))
    if a.output.exists():raise FileExistsError(a.output)
    save_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
