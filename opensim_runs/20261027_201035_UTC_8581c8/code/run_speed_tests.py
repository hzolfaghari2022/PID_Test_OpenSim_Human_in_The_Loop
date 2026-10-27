"""Fixed-gain OpenSim speed tests. No NN, learning or retuning. Upload completed runs to GitHub by default."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
# Avoid excessive BLAS/OpenMP threading in these small simulation models.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SCHEDULES = {
    'slow': [4., 4.] * 4,
    'original_speed': [3., 3.] * 4,
    'medium': [2., 2.] * 4,
    'fast': [1., 1.] * 4,
    'changing_speed': [4., 4., 2., 2., 1., 1., 3., 3.],
}

def target(t, legs):
    elapsed = t - 0.5
    if elapsed <= 0:
        return math.radians(20.)
    for i, duration in enumerate(legs):
        if elapsed < duration:
            u = elapsed / duration
            smooth = 10*u**3 - 15*u**4 + 6*u**5
            return math.radians(20 + 60*(smooth if i % 2 == 0 else 1-smooth))
        elapsed -= duration
    return math.radians(20.)

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def metrics(rows, legs, tmax):
    motion = [r for r in rows if .5 <= float(r['time_s']) < .5 + sum(legs)]
    e = [float(r['error_deg']) for r in motion]
    f = [float(r['tension_N']) for r in motion]
    return {
        'motion_rmse_deg': math.sqrt(sum(v*v for v in e)/len(e)),
        'motion_mae_deg': sum(abs(v) for v in e)/len(e),
        'motion_peak_abs_error_deg': max(map(abs,e)),
        'motion_peak_tension_N': max(f),
        'upper_limit_percent': 100*sum(v >= tmax-1e-6 for v in f)/len(f),
        'zero_tension_percent': 100*sum(v <= 1e-6 for v in f)/len(f),
        'any_clipping_percent':100*sum(int(float(r['saturated'])) for r in motion)/len(motion),
    }

def trial_plot(plt, rows, dest, name):
    t=[float(r['time_s']) for r in rows]
    fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
    ax[0].plot(t,[float(r['reference_deg']) for r in rows],'k--',label='Reference')
    ax[0].plot(t,[float(r['angle_deg']) for r in rows],label='Simulated angle')
    ax[0].set_ylabel('Elbow angle (deg)');ax[0].legend()
    ax[1].plot(t,[float(r['error_deg']) for r in rows]);ax[1].set_ylabel('Error (deg)')
    ax[2].plot(t,[float(r['tension_N']) for r in rows]);ax[2].set_ylabel('Cable tension (N)');ax[2].set_xlabel('Time (s)')
    for a in ax:a.grid(alpha=.25)
    fig.suptitle(name+' | fixed gains');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(dest/('tracking.'+ext),dpi=180)
    plt.close(fig)

def worker(config_path):
    cfg=json.loads(Path(config_path).read_text())
    dest=Path(cfg['output']);dest.mkdir(parents=True,exist_ok=True)
    status={'experiment':cfg['name'],'status':'failed','results_folder':str(dest)}
    if (dest.parent/'INTERRUPTED_FOR_RUNTIME_CONFIGURATION.txt').exists():
        status['status']='interrupted'
        (dest/'status.json').write_text(json.dumps(status,indent=2))
        return 1
    try:
        import opensim as osim
        import matplotlib
        if not cfg['live']:matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        module=load_module('original_runner',ROOT/'Python'/'run_arm26_pi_github.py')
        # Override the reference only. Retain the supplied control and integration law.
        module.reference=lambda t,unused:target(t,cfg['legs'])
        module.DEMO_CYCLES=len(cfg['legs'])//2
        osim.Logger.setLevelString('Warn')
        source=Path(cfg['model'])
        args=SimpleNamespace(kp=cfg['kp'],ki=cfg['ki'],kv=cfg['kv'],tmax=cfg['tmax'],
            dt=cfg['dt'],duration=.5+sum(cfg['legs'])+1.,rise_time=cfg['legs'][0],
            live=cfg['live'],plt=plt,results_dir=dest)
        module.prepare_display(osim,source,ROOT/'Geometry',False)
        if args.live:
            try:module.prepare_display(osim,source,ROOT/'Geometry',True)
            except Exception as exc:
                print('Viewer unavailable; continuing without live display:',exc,flush=True)
                args.live=False
        rows,_=module.run_trial(osim,source,dest,args,args.ki,'PI')
        trial_plot(plt,rows,dest,cfg['name'])
        status.update(metrics(rows,cfg['legs'],cfg['tmax']))
        status.update(status='completed',opensim_version=osim.GetVersion())
        if hasattr(args,'display'):
            plt.show();args.display.close()
    except Exception as exc:
        status['error']=str(exc)
        traceback.print_exc()
    (dest/'status.json').write_text(json.dumps(status,indent=2))
    return 0 if status['status']=='completed' else 1

def summary_plot(suite, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    done=[r for r in results if r['status']=='completed']
    if not done:return
    fig,ax=plt.subplots(len(done),2,figsize=(12,2.6*len(done)),squeeze=False)
    for i,result in enumerate(done):
        with (Path(result['results_folder'])/'PI_data.csv').open() as handle:rows=list(csv.DictReader(handle))
        t=[float(r['time_s']) for r in rows]
        ax[i,0].plot(t,[float(r['reference_deg']) for r in rows],'k--',label='Reference')
        ax[i,0].plot(t,[float(r['angle_deg']) for r in rows],label='Angle')
        ax[i,0].set_ylabel(result['experiment']+'\nangle (deg)')
        ax[i,1].plot(t,[float(r['tension_N']) for r in rows]);ax[i,1].set_ylabel('Tension (N)')
        for a in ax[i]:a.grid(alpha=.25);a.set_xlabel('Time (s)')
    ax[0,0].legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(suite/('speed_comparison.'+ext),dpi=180)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--mass',type=float,default=0.,help='Dumbbell mass in kg; default 0 means unloaded.')
    parser.add_argument('--model',type=Path,help='Optional existing model. Do not combine with nonzero --mass.')
    parser.add_argument('--kp',type=float,default=30.)
    parser.add_argument('--ki',type=float,default=15.)
    parser.add_argument('--kv',type=float,default=0.,help='Velocity damping, not reference-error derivative.')
    parser.add_argument('--tmax',type=float,default=80.)
    parser.add_argument('--dt',type=float,default=.01)
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--no-push',action='store_true',help='Save locally without uploading to GitHub.')
    parser.add_argument('--github-ssh',action='store_true',help='Use your configured GitHub SSH key instead of HTTPS credentials.')
    parser.add_argument('--no-open',action='store_true',help='Save figures without opening the results gallery.')
    parser.add_argument('--only',choices=list(SCHEDULES),help='Run one experiment only.')
    args=parser.parse_args()
    if args.worker:return worker(args.worker)
    values=[args.mass,args.kp,args.ki,args.kv,args.tmax,args.dt]
    if not all(math.isfinite(v) for v in values) or min(values[:4])<0 or min(values[4:])<=0:
        parser.error('Use finite nonnegative mass/gains and positive force limit/time step.')
    if args.model and args.mass:parser.error('Use either --model or --mass, not both.')
    for duration in [0.5,1.,2.,3.,4.]:
        if abs(duration/args.dt-round(duration/args.dt))>1e-8:parser.error('--dt must divide 0.5, 1, 2, 3 and 4 seconds exactly.')
    try:import opensim as osim
    except ImportError:
        raise SystemExit('OpenSim is unavailable in this Python. Activate your working lstm-py312 environment first.')
    suite=ROOT/'speed_test_results'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC_')+uuid.uuid4().hex[:6])
    suite.mkdir(parents=True)
    source=args.model.expanduser().resolve() if args.model else ROOT/'arm26_exosuit.osim'
    if not source.is_file():raise FileNotFoundError(source)
    if args.mass:
        helper=load_module('dumbbell_helper',ROOT/'add_dumbbell.py')
        osim.ModelVisualizer.addDirToGeometrySearchPaths(str(ROOT/'Geometry'))
        helper.save_loaded_model(osim,source,suite/'loaded_model.osim',args.mass)
        source=suite/'loaded_model.osim'
    results=[]
    for name,legs in SCHEDULES.items():
        if args.only and name!=args.only:continue
        dest=suite/name;dest.mkdir()
        cfg=dict(name=name,legs=legs,model=str(source),output=str(dest),
                 mass_kg=args.mass if not args.model else 'specified by external model',
                 kp=args.kp,ki=args.ki,kv=args.kv,tmax=args.tmax,dt=args.dt,live=args.live)
        config=dest/'settings.json';config.write_text(json.dumps(cfg,indent=2))
        print('\n========== '+name.upper()+' ==========',flush=True)
        with (dest/'console.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(config)],
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=proc.wait()
        status=dest/'status.json'
        result=json.loads(status.read_text()) if status.exists() else dict(experiment=name,status='failed',results_folder=str(dest),error='Worker exited without status')
        if code!=0:result['status']='failed'
        results.append(result)
        (suite/'summary.json').write_text(json.dumps(results,indent=2))
        fields=sorted({k for r in results for k in r})
        with (suite/'summary.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(results)
    try:summary_plot(suite,results)
    except Exception:traceback.print_exc();print('Summary plots unavailable; CSV/JSON remain saved.')
    try:
        from analyze_speed_results import analyze
        analyze(suite)
    except Exception:
        traceback.print_exc();print('Diagnostics unavailable; original outputs remain saved.')
    print('\nRESULTS:',suite)
    try:
        from open_results import build_gallery, open_gallery
        if args.no_open:build_gallery(suite)
        else:open_gallery(suite)
    except Exception as exc:
        print('Figure gallery unavailable:',exc,'; saved figures remain in',suite)
    push_ok=True
    if not args.no_push:
        from push_speed_results import upload
        push_ok=upload(suite,source=source,ssh=args.github_ssh)
    else:
        (suite/'github_push_status.txt').write_text('Upload disabled by --no-push.\n')
    return int(not push_ok or any(r['status']!='completed' for r in results))

if __name__=='__main__':
    raise SystemExit(main())
