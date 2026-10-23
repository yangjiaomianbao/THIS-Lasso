"""Uniform certified full-LASSO backend for all five-system validation models."""
import argparse
import concurrent.futures
import importlib.util
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from threadpoolctl import threadpool_limits
from src.bridge.numeric_io import file_hash,write_toml
from src.solvers.lars_validation import validation_lasso_accelerated
name='dynamics15_parallel_runtime'
spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/dynamics15_parallel.py')
scheduler=importlib.util.module_from_spec(spec);sys.modules[name]=scheduler;spec.loader.exec_module(scheduler)
r=scheduler.r;pilot=r.pilot
ORIGINAL_INITIALIZER=r.initialize


def initialize(data,c,root,trial,method):
    ORIGINAL_INITIALIZER(data,c,root,trial,{'M1vC':'M1vW','M3C':'M3W'}.get(method,method))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',default='five_systems_certified_20261022')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/dynamics15_five_systems_certified_v1_1.toml')
    parser.add_argument('--make-config',action='store_true')
    args=parser.parse_args();assert Path(args.label).name==args.label and args.label not in ('.','..')
    if args.make_config:
        assert not args.config.exists()
        c=tomllib.loads((ROOT/'configs/dynamics15_five_systems_v1.toml').read_text())
        c.update(experiment='dynamics15_five_systems_full_cubic_certified_v1_1',
            method_order=['M0D','M0sD','M1B','M2B','M1vC','M3C'],
            numerical_backend='All validation models: uniform full weighted LARS + independent all-grid KKT; original working-set fallback',
            lars_max_knots=5000)
        write_toml(args.config,c);return
    c=tomllib.loads(args.config.read_text());out=ROOT/'results/dynamics15'/args.label
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();(out/'julia_depot').mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        JULIA_DEPOT_PATH=str(out/'julia_depot')+':'+str(ROOT/'.julia-depot-r0'))
    r.initialize=initialize;r.validation_lasso_working_set=validation_lasso_accelerated
    started=time.perf_counter()
    try:
        pilot.save_json(out/'integrity_before.json',r.integrity());(out/'config.toml').write_bytes(args.config.read_bytes())
        sources=['scripts/dynamics15_accelerated.py','scripts/dynamics15.py','scripts/dynamics15_parallel.py',
            'scripts/dynamics15_prepare.jl','src/data/multisystem_dynamics.jl','src/data/paper_dynamics.jl',
            'src/solvers/lars_validation.py','src/solvers/large_library.py','src/solvers/adaptive_lasso.py',
            'src/solvers/plain_lasso.py','src/solvers/methods.py','scripts/dynamics30.py','scripts/dynamics_pilot.py','src/metrics/structure.py']
        pilot.save_json(out/'protocol.json',dict(config=c,source_sha256={name:file_hash(ROOT/name) for name in sources},
            candidate_screening=False,truth_used_for_training=False,selection_locked_before_test=True,
            sources={s:c[s]['source'] for s in c['systems']}))
        pilot.save_json(out/'execution_accelerator.json',dict(uniform_backend_for_all_validation_models=True,
            validation_models_refitted=40,models_reused=0,original_objective_grid_data_seeds_tolerances_unchanged=True,
            previous_unscored_run='../five_systems_full_20261022',truth_or_test_used_for_acceleration=False,
            source_sha256={name:file_hash(ROOT/name) for name in ['scripts/dynamics15_accelerated.py','src/solvers/lars_validation.py','tests/test_lars_validation.py']},
            benchmark_file='../five_systems_full_20261022/logs/lars_weighted_benchmark.json',
            numerical_endpoint_policy='Uniform coordinate/alpha rescaling preserves objective; every actual alpha must pass original-unit global KKT; otherwise original working-set fallback'))
        runtime=['/home/yg/.local/opt/julia-1.9.4/bin/julia','--startup-file=no','--project='+str(ROOT/'environment/r0/julia')]
        for tag,command in [('dynamics_tests',runtime+[str(ROOT/'tests/test_multisystem_dynamics.jl')]),
            ('prepare',runtime+[str(ROOT/'scripts/dynamics15_prepare.jl'),str(args.config),str(out)])]:
            with (out/'logs'/f'julia_{tag}.log').open('x') as log:result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Julia {tag} failed')
        with threadpool_limits(limits=1):
            with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork')) as pool:
                jobs=[pool.submit(scheduler._dataset,c,out,s,seed) for s,seed in itertools.product(c['systems'],c['seeds'])]
                for job in concurrent.futures.as_completed(jobs):print('DATASET_COMPLETE',*job.result(),flush=True)
            locks={}
            for s,seed,ratio,method in itertools.product(c['systems'],c['seeds'],c['noise_ratios'],c['method_order']):
                trial=out/s/f'seed_{seed}'/pilot.slot(ratio)/method
                locks[str(trial.relative_to(out))]=dict(status_sha256=file_hash(trial/'status.json'),
                    coefficients_sha256=file_hash(trial/'coefficients.npz') if (trial/'coefficients.npz').exists() else 'UNAVAILABLE')
            pilot.save_json(out/'selection_locked.json',dict(trials=locks,test_evaluated_before_lock=False))
            records,summary=r.d30.evaluate(c,out)
        for name,expected in locks.items():
            assert file_hash(out/name/'status.json')==expected['status_sha256']
            if expected['coefficients_sha256']!='UNAVAILABLE':assert file_hash(out/name/'coefficients.npz')==expected['coefficients_sha256']
        pilot.save_json(out/'integrity_after.json',r.integrity())
        states=[a['status'] for a in records if a['order']=='triple']
        complete=dict(status='COMPLETE_WITH_FAILURES' if 'FAILED' in states else 'COMPLETE',models=len(states),
            counts={s:states.count(s) for s in ('OK','NA','FAILED')},systems=c['systems'],physical_nodes=15,
            pair_edges=45,triple_edges=15,wall_seconds=time.perf_counter()-started,uniform_certified_backend=True)
        pilot.save_json(out/'completion.json',complete);print(json.dumps(complete),flush=True)
    except Exception:
        pilot.save_json(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
