"""Resumable OpenSim PID demonstrations. No neural network or EWC training.
Every attempted episode and every sampled physical state are retained.
Run from any directory; edit collection_config.json, then use START_COLLECTION.bat.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import webbrowser
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parent
MUSCLES = ['TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA']
FEATURES = ['angle_rad', 'velocity_rad_s', 'reference_rad', 'reference_velocity_rad_s',
            'reference_acceleration_rad_s2', 'mass_kg', 'previous_tension_N', 'I_Nm']
FEATURES += [m + '_activation' for m in MUSCLES]
FEATURES += [m + '_normalized_fiber_length' for m in MUSCLES]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(t, cfg):
    elapsed = t - cfg['start_hold_s']
    lo, hi, T = math.radians(cfg['start_deg']), math.radians(cfg['target_deg']), cfg['leg_duration_s']
    if elapsed <= 0 or elapsed >= 2 * cfg['cycles'] * T:
        return lo, 0., 0., False
    leg = min(int(elapsed / T), 2 * cfg['cycles'] - 1)
    u = (elapsed - leg * T) / T
    s = 10*u**3 - 15*u**4 + 6*u**5
    d = (hi-lo) * (1 if leg % 2 == 0 else -1)
    return ((lo if leg % 2 == 0 else hi) + d*s,
            d*(30*u*u-60*u**3+30*u**4)/T,
            d*(60*u-180*u*u+120*u**3)/T**2, True)


def build(o, cfg):
    from add_dumbbell import add_dumbbell, DEFAULT_GRIP
    o.ModelVisualizer.addDirToGeometrySearchPaths(str(ROOT/'Geometry'))
    model = o.Model(str(ROOT/'arm26_exosuit.osim'))
    if model.getControllerSet().getSize():
        raise ValueError('Expected original model without embedded controllers.')
    add_dumbbell(o, model, cfg['mass_kg'], DEFAULT_GRIP)
    model.setGravity(o.Vec3(0, -9.80665, 0))
    if [model.getCoordinateSet().get(i).getName() for i in range(model.getCoordinateSet().getSize())] != ['r_shoulder_elev', 'r_elbow_flex']:
        raise ValueError('Unexpected coordinate order; inverse-dynamics indexing must be reviewed.')
    el = model.updCoordinateSet().get('r_elbow_flex')
    sh = model.updCoordinateSet().get('r_shoulder_elev')
    sh.setDefaultValue(0); sh.setDefaultSpeedValue(0); sh.setDefaultLocked(True)
    el.setDefaultValue(math.radians(cfg['start_deg'] + cfg['initial_offset_deg']))
    el.setDefaultSpeedValue(0); el.setDefaultLocked(False)
    ctrl = o.PrescribedController(); ctrl.setName('unchanged_fixed_muscle_excitation')
    mus = model.updMuscles()
    actual_names = [mus.get(i).getName() for i in range(mus.getSize())]
    if actual_names != MUSCLES:
        raise ValueError('Muscle order differs from the documented feature layout: ' + str(actual_names))
    for i in range(mus.getSize()):
        m = mus.get(i)
        m.setMaxIsometricForce(m.getMaxIsometricForce()*cfg['strength_scale'])
        o.Thelen2003Muscle.safeDownCast(m).setDefaultActivation(cfg['muscle_excitation'])
        ctrl.addActuator(m)
        ctrl.prescribeControlForActuator(m.getName(), o.Constant(cfg['muscle_excitation']))
    model.addController(ctrl); model.setUseVisualizer(False)
    state = model.initSystem(); state.setTime(0); model.equilibrateMuscles(state)
    cable = o.PathActuator.safeDownCast(model.updForceSet().get('exo_flexor'))
    if cable is None:
        raise ValueError('Missing exo_flexor PathActuator.')
    cable.overrideActuation(state, True); cable.setOverrideActuation(state, 0)
    return model, state, el, cable


def metrics(rows, cfg):
    def group(rs):
        if not rs:
            return None
        e = np.array([r['error_deg'] for r in rs]); f = np.array([r['tension_N'] for r in rs])
        return dict(rmse_deg=float(np.sqrt(np.mean(e*e))), mae_deg=float(np.mean(abs(e))),
                    max_abs_error_deg=float(max(abs(e))), peak_tension_N=float(max(f)),
                    mean_tension_N=float(np.mean(f)),
                    upper_saturation_percent=100*float(np.mean(f >= .999*cfg['force_cap_N'])),
                    clipped_percent=100*float(np.mean([r['saturated'] for r in rs])),
                    peak_abs_assistance_torque_Nm=float(max(abs(r['assistance_torque_Nm']) for r in rs)),
                    peak_actual_hand_speed_m_s=float(max(r['actual_hand_speed_m_s'] for r in rs)))
    return dict(all_samples=group(rows), motion_only=group([r for r in rows if r['moving']]))


def plot_episode(rows, dest, cfg, status):
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11})
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t = [r['time_s'] for r in rows]
    ax[0].plot(t, [r['reference_deg'] for r in rows], 'k--', label='Desired')
    ax[0].plot(t, [r['angle_deg'] for r in rows], color='#0072b2', label='PID actual')
    ax[0].set_ylabel('Elbow angle (deg)'); ax[0].legend()
    ax[1].plot(t, [r['error_deg'] for r in rows], color='#d55e00'); ax[1].set_ylabel('Error (deg)')
    ax[2].plot(t, [r['tension_N'] for r in rows], color='#009e73')
    ax[2].axhline(cfg['force_cap_N'], color='black', ls=':', label='Cable limit')
    ax[2].set(ylabel='Cable tension (N)', xlabel='Time (s)'); ax[2].legend()
    for a in ax:
        a.grid(alpha=.2)
    fig.suptitle(f"{cfg['episode_id']} | {cfg['mass_kg']:g} kg | peak hand speed {cfg['peak_hand_speed_m_s']:.3g} m/s\n"
                 f"Kp={cfg['kp']:g}, Ki={cfg['ki']:g}, Kd={cfg['kd']:g} | {status}")
    fig.tight_layout(); fig.savefig(dest/'tracking.png', dpi=150); plt.close(fig)


def run_episode(dest):
    """One process per episode; preserve partial CSV even on interruption/failure."""
    import opensim as o
    from add_dumbbell import DEFAULT_GRIP
    o.Logger.setLevelString('Error')
    dest = Path(dest); cfg = read_json(dest/'settings.json')
    started = time.perf_counter(); rows = []; model = None
    result = dict(episode_id=cfg['episode_id'], task=cfg['task'], split=cfg['split'],
                  status='running', accepted_for_imitation=False, error='')
    write_json(dest/'status.json', result)
    try:
        model, state, el, cable = build(o, cfg)
        # A locked shoulder and one moving elbow give a constant station radius.
        probe = o.State(state); el.setSpeedValue(probe, 1.0); model.realizeVelocity(probe)
        v = model.getBodySet().get('r_ulna_radius_hand').findStationVelocityInGround(probe, o.Vec3(*DEFAULT_GRIP))
        radius = math.sqrt(sum(v.get(i)**2 for i in range(3)))
        T = 1.875*math.radians(cfg['target_deg']-cfg['start_deg'])*radius/cfg['peak_hand_speed_m_s']
        cfg['leg_duration_s'] = T; cfg['hand_radius_m'] = radius
        end = cfg['start_hold_s']+2*cfg['cycles']*T+cfg['end_hold_s']
        cfg['duration_s'] = end; cfg['opensim_version'] = o.GetVersion()
        model.realizeDynamics(state)
        # Optional documented integral preload, not a runtime feedforward controller.
        I = 0.
        if cfg['preload_integral'] and cfg['ki'] > 0:
            residual = o.InverseDynamicsSolver(model).solve(state, o.Vector(model.getNumSpeeds(), 0))
            arm = cable.computeMomentArm(state, el)
            I = float(np.clip(residual.get(1), 0, cfg['force_cap_N']*arm))
        cfg['initial_integral_torque_Nm'] = I
        write_json(dest/'settings.json', cfg)
        model.setName('arm26_pid_dataset_playback'); model.printToXML(str(dest/'playback_model.osim'))
        mus = model.getMuscles()
        names_arr = model.getStateVariableNames()
        state_names = [names_arr.get(i) for i in range(names_arr.getSize())]
        write_json(dest/'state_names.json', state_names)
        previous = 0.
        with (dest/'signals.csv').open('w', newline='', encoding='utf-8', buffering=1) as signal_file, \
             (dest/'states.csv').open('w', newline='', encoding='utf-8', buffering=1) as state_file:
            sw = None; stw = csv.writer(state_file); stw.writerow(['time_s']+state_names)
            steps = math.ceil(end/cfg['dt_s']); dt = cfg['dt_s']
            for k in range(steps+1):
                t = float(state.getTime()); h = min(dt, max(0., end-t))
                model.realizeVelocity(state)
                q, qd = el.getValue(state), el.getSpeedValue(state)
                qr, vr, ar, moving = reference(t, cfg)
                error = qr-q; arm = cable.computeMomentArm(state, el)
                if not 0 <= math.degrees(q) <= 90 or arm <= .005:
                    raise RuntimeError(f'Outside checked geometry: t={t:g}, q={math.degrees(q):g}, moment arm={arm:g}')
                P = cfg['kp']*error; D = cfg['kd']*(vr-qd)
                raw = P+I+D; F = float(np.clip(raw/arm, 0, cfg['force_cap_N']))
                cable.setOverrideActuation(state, F); model.realizeDynamics(state)
                actual = cable.getActuation(state)
                if abs(actual-F) > 1e-6:
                    raise RuntimeError('Applied cable force differs from requested force.')
                gripv = model.getBodySet().get('r_ulna_radius_hand').findStationVelocityInGround(state, o.Vec3(*DEFAULT_GRIP))
                r = dict(time_s=t, episode_id=cfg['episode_id'], task=cfg['task'], split=cfg['split'],
                         mass_kg=cfg['mass_kg'], strength_scale=cfg['strength_scale'],
                         requested_peak_hand_speed_m_s=cfg['peak_hand_speed_m_s'],
                         reference_deg=math.degrees(qr), angle_deg=math.degrees(q), error_deg=math.degrees(error),
                         angle_rad=q, velocity_rad_s=qd, reference_rad=qr, reference_velocity_rad_s=vr,
                         reference_acceleration_rad_s2=ar, reference_hand_speed_m_s=radius*abs(vr),
                         actual_hand_speed_m_s=math.sqrt(sum(gripv.get(i)**2 for i in range(3))), moving=int(moving),
                         previous_tension_N=previous, tension_N=actual, force_cap_N=cfg['force_cap_N'],
                         moment_arm_m=arm, assistance_torque_Nm=actual*arm, raw_torque_Nm=raw,
                         P_Nm=P, I_Nm=I, D_Nm=D, Kp=cfg['kp'], Ki=cfg['ki'], Kd=cfg['kd'],
                         saturated=int(abs(raw-actual*arm)>1e-8), muscle_excitation=cfg['muscle_excitation'])
                mtau = 0.
                for i, name in enumerate(MUSCLES):
                    m = mus.get(i); tendon = m.getTendonForce(state); moment = m.computeMomentArm(state, el)
                    r[name+'_activation'] = m.getActivation(state)
                    r[name+'_fiber_length_m'] = m.getFiberLength(state)
                    r[name+'_normalized_fiber_length'] = m.getFiberLength(state)/m.getOptimalFiberLength()
                    r[name+'_fiber_velocity_m_s'] = m.getFiberVelocity(state)
                    r[name+'_path_length_m'] = m.getLength(state)
                    r[name+'_tendon_force_N'] = tendon
                    r[name+'_moment_arm_m'] = moment
                    r[name+'_torque_Nm'] = tendon*moment; mtau += tendon*moment
                r['muscle_torque_Nm'] = mtau
                if not all(math.isfinite(v) for v in r.values() if isinstance(v, (int,float))):
                    raise FloatingPointError('Nonfinite simulation record.')
                if sw is None:
                    sw = csv.DictWriter(signal_file, fieldnames=list(r)); sw.writeheader()
                sw.writerow(r); rows.append(r)
                sv = model.getStateVariableValues(state)
                stw.writerow([t]+[sv.get(i) for i in range(sv.size())])
                if k % max(1, round(.5/dt)) == 0:
                    signal_file.flush(); state_file.flush()
                if k == steps:
                    break
                if not ((raw >= cfg['force_cap_N']*arm and error > 0) or (raw <= 0 and error < 0)):
                    I += cfg['ki']*error*h
                manager = o.Manager(model); manager.setIntegratorAccuracy(cfg['integrator_accuracy'])
                manager.setIntegratorMaximumStepSize(h/5); manager.initialize(state)
                state = o.State(manager.integrate(t+h)); previous = actual
        result['status'] = 'completed'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        (dest/'error.txt').write_text(result['error'], encoding='utf-8')
        if model is not None and 'state' in locals():
            try:
                sv=model.getStateVariableValues(state)
                values=[sv.get(i) for i in range(sv.size())]
                write_json(dest/'failure_terminal_state.json',dict(time_s=float(state.getTime()),
                           state_values=[v if math.isfinite(v) else str(v) for v in values]))
            except Exception:pass
    finally:
        result.update(metrics(rows, cfg)); m = result.get('motion_only')
        result['accepted_for_imitation'] = bool(result['status']=='completed' and m and
            m['rmse_deg']<=cfg['accept_rmse_deg'] and result['all_samples']['max_abs_error_deg']<=cfg['accept_max_error_deg'])
        result['sample_count'] = len(rows); result['wall_seconds'] = time.perf_counter()-started
        result['last_recorded_time_s'] = rows[-1]['time_s'] if rows else 0
        try:
            if rows:
                X = np.array([[r[n] for n in FEATURES] for r in rows], dtype=np.float64)
                Y = np.array([[r['tension_N']] for r in rows], dtype=np.float64)
                np.savez_compressed(dest/'training_arrays.npz', X_raw=X, y_force_N=Y,
                                    feature_names=np.array(FEATURES), time_s=np.array([r['time_s'] for r in rows]))
                with (dest/'states.csv').open(encoding='utf-8') as f:
                    sr=list(csv.reader(f))
                with (dest/'states.sto').open('w',encoding='utf-8') as f:
                    f.write(f"Arm26 PID states\nversion=1\nnRows={len(sr)-1}\nnColumns={len(sr[0])}\ninDegrees=no\nendheader\n")
                    f.write('\t'.join(['time']+sr[0][1:])+'\n')
                    for rr in sr[1:]:f.write('\t'.join(rr)+'\n')
                if cfg.get('save_episode_plots', True):
                    plot_episode(rows, dest, cfg, 'ACCEPTED' if result['accepted_for_imitation'] else 'NOT ACCEPTED')
        except Exception:
            result['export_error'] = traceback.format_exc()
            result['accepted_for_imitation'] = False
        write_json(dest/'status.json', result)
    return 0 if result['status']=='completed' and 'export_error' not in result else 1


def validate(c):
    for n in ['episodes_per_task','cycles','seed']:
        if not isinstance(c[n],int) or c[n] < (0 if n=='seed' else 1):
            raise ValueError(n+' must be an integer in its allowed range.')
    for n in ['dt_s','force_cap_N','integrator_accuracy','accept_rmse_deg','accept_max_error_deg']:
        if not math.isfinite(c[n]) or c[n]<=0:
            raise ValueError(n+' must be finite and positive.')
    if not 0 < c['strength_scale'] <= 1 or not 0 <= c['muscle_excitation'] <= 1:
        raise ValueError('Invalid muscle strength or excitation.')
    for n in ['kp','ki','kd','start_hold_s','end_hold_s']:
        if not math.isfinite(c[n]) or c[n]<0:raise ValueError('Invalid '+n)
    if c['dt_s']>.02 or c['dt_s']<1e-5:raise ValueError('Use dt between 0.00001 and 0.02 s.')
    for task in c['tasks']:
        if not task['name'].isalnum() or not all(math.isfinite(task[n]) for n in ['mass_kg','peak_hand_speed_m_s']) or task['mass_kg']<0 or task['peak_hand_speed_m_s']<=0:
            raise ValueError('Invalid task.')
    if len({t['name'] for t in c['tasks']}) != len(c['tasks']):raise ValueError('Duplicate task names.')
    for n in ['start_deg_range','initial_offset_deg_range','speed_factor_range','mass_factor_range']:
        a,b=c.get(n,[1.,1.])
        if not math.isfinite(a) or not math.isfinite(b) or a>b:
            raise ValueError('Invalid range: '+n)
        if n in ['speed_factor_range','mass_factor_range'] and a<=0:
            raise ValueError('Factors must be positive: '+n)
    if not math.isfinite(c['episode_timeout_s']) or c['episode_timeout_s']<=0:
        raise ValueError('Episode timeout must be positive.')


def plan(c):
    rng = np.random.default_rng(c['seed']); episodes=[]
    for task in c['tasks']:
        n=c['episodes_per_task']; ix=rng.permutation(n)
        splits={int(j):('train' if k<int(.7*n) else 'validation' if k<int(.85*n) else 'test') for k,j in enumerate(ix)}
        for i in range(n):
            e=dict(c); e.pop('tasks'); e.update(task)
            e.update(task=task['name'], episode_id=f"{task['name']}_{i+1:04d}", split=splits[i])
            e.pop('name')
            e['start_deg'] = c['start_deg'] if i==0 else float(rng.uniform(*c['start_deg_range']))
            e['initial_offset_deg'] = 0. if i==0 else float(rng.uniform(*c['initial_offset_deg_range']))
            factor=1. if i==0 else float(rng.uniform(*c['speed_factor_range']))
            e['peak_hand_speed_m_s'] *= factor
            e['mass_kg'] *= 1. if i==0 else float(rng.uniform(*c.get('mass_factor_range',[1.,1.])))
            if not 0 < e['start_deg']+e['initial_offset_deg'] < e['target_deg'] < 90:
                raise ValueError('Requested initial/reference angles leave checked geometry.')
            episodes.append(e)
    return episodes


def summarize(suite):
    episodes=read_json(suite/'episode_plan.json'); records=[]
    for e in episodes:
        p=suite/e['episode_id']/'status.json'
        if not p.exists():continue
        r=read_json(p);m=r.get('motion_only') or {}
        records.append(dict(episode_id=e['episode_id'],task=e['task'],split=e['split'],
                            mass_kg=e['mass_kg'],peak_hand_speed_m_s=e['peak_hand_speed_m_s'],
                            status=r['status'],accepted_for_imitation=r.get('accepted_for_imitation',False),
                            sample_count=r.get('sample_count',0),rmse_deg=m.get('rmse_deg'),
                            max_error_deg=(r.get('all_samples') or {}).get('max_abs_error_deg'),
                            upper_saturation_percent=m.get('upper_saturation_percent')))
    write_json(suite/'summary.json',records)
    if records:
        with (suite/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    counts={t['name']:dict(attempted=sum(r['task']==t['name'] for r in records),
                         accepted=sum(r['task']==t['name'] and r['accepted_for_imitation'] for r in records))
            for t in read_json(suite/'collection_config.json')['tasks']}
    write_json(suite/'counts.json',counts)
    table=[]
    for r in records:
        metric='--' if r['rmse_deg'] is None else f"{r['rmse_deg']:.2f}"
        table.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['episode_id'],r['task'],r['split'],r['status'],r['accepted_for_imitation'],metric])+
                     f"<td><a href='{r['episode_id']}/tracking.png'>Plot</a> | <a href='{r['episode_id']}/signals.csv'>All signals</a></td></tr>")
    page='<html><head><meta charset="utf-8"><title>PID collection</title><style>body{font:16px Arial;margin:35px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style></head><body><h1>PID data collection</h1>'
    page+='<p>Accepted = passed declared tracking criteria. All other records are still saved.</p>'
    page+='<pre>'+html.escape(json.dumps(counts,indent=2))+'</pre><table><tr><th>Episode</th><th>Task</th><th>Split</th><th>Simulation</th><th>Accepted</th><th>Motion RMSE (deg)</th><th>Files</th></tr>'+''.join(table)+'</table></body></html>'
    (suite/'VIEW_RESULTS.html').write_text(page,encoding='utf-8')
    return counts


def publish(suite):
    """Frozen per-episode upload; no GUI and no credential prompt can stall collection."""
    import original_pi_demo as support
    support.GITHUB_REPO_URL='https://github.com/hzolfaghari2022/PID_Test_OpenSim_Human_in_The_Loop.git'
    support.GITHUB_RESULTS_FOLDER='pid_data_collection/'+suite.parent.name
    code=suite/'source';code.mkdir(exist_ok=True)
    for name in ['collect_pid_data.py','export_training_data.py','add_dumbbell.py','arm26_exosuit.osim','original_pi_demo.py']:
        source=suite.parent/'source'/name
        if source.is_file() and not (code/name).exists():shutil.copy2(source,code/name)
    support.GITHUB_BRANCH='main'
    files=tuple(sorted(p.relative_to(suite).as_posix() for p in suite.rglob('*') if p.is_file()
                       and p.name not in ['upload_manifest.json','github_push_status.txt']))
    if any((suite/n).stat().st_size>=95*1024*1024 for n in files):
        raise ValueError('A file is too large for ordinary Git upload.')
    support.GITHUB_FILES=files
    write_json(suite/'upload_manifest.json',dict(run_id=suite.name,files={n:sha(suite/n) for n in files}))
    # Override helper to prohibit interactive authentication and set a timeout.
    def git_command(git,*args,cwd=None):
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
        p=subprocess.run([git,*args],cwd=cwd,env=env,text=True,capture_output=True,timeout=60)
        if p.returncode:raise RuntimeError(f'Git {args[0]} failed; authenticate Git on this computer, then retry upload.')
        return p.stdout.strip()
    support.git_command=git_command
    return support.push_completed_run(suite)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'collection_config.json')
    p.add_argument('--resume',type=Path);p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--episodes',type=int,help='Override episodes per task for a new collection only.')
    p.add_argument('--no-push',action='store_true');p.add_argument('--no-open',action='store_true')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--push-results',type=Path,help='Retry upload of one saved episode folder.')
    args=p.parse_args()
    if args.worker:return run_episode(args.worker)
    if args.push_results:return 0 if publish(args.push_results.resolve()) else 1
    if args.resume:
        suite=args.resume.resolve();c=read_json(suite/'collection_config.json')
        old=read_json(suite/'provenance.json')
        if old['collector_sha256']!=sha(__file__) or old['input_model_sha256']!=sha(ROOT/'arm26_exosuit.osim') or old['dumbbell_helper_sha256']!=sha(ROOT/'add_dumbbell.py'):
            raise ValueError('Code/model changed since collection began. Start a new collection.')
        episodes=read_json(suite/'episode_plan.json')
    else:
        c=read_json(args.config)
        if args.episodes is not None:c['episodes_per_task']=args.episodes
        validate(c);episodes=plan(c)
        suite=ROOT/'pid_datasets'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')+'_'+os.urandom(3).hex())
        suite.mkdir(parents=True)
        write_json(suite/'collection_config.json',c);write_json(suite/'episode_plan.json',episodes)
        write_json(suite/'provenance.json',dict(collector_sha256=sha(__file__),input_model_sha256=sha(ROOT/'arm26_exosuit.osim'),
                   dumbbell_helper_sha256=sha(ROOT/'add_dumbbell.py'),python=sys.version,
                   features=FEATURES,normalization='RAW physical units; fit preprocessing on Task A training records only.',
                   target='Applied PID cable tension, N',notes='Fixed muscle excitation; no physiological normal/weak assistance claim.'))
        source=suite/'source';source.mkdir()
        for n in ['collect_pid_data.py','export_training_data.py','add_dumbbell.py','arm26_exosuit.osim','collection_config.json','original_pi_demo.py']:
            shutil.copy2(ROOT/n,source/n)
        write_json(source/'collection_config.json',c)
    (ROOT/'LATEST_DATASET.txt').write_text(str(suite),encoding='utf-8')
    print('DATASET:',suite,flush=True)
    print(f"{len(episodes)} episodes; Kp={c['kp']}, Ki={c['ki']}, Kd={c['kd']}; force cap={c['force_cap_N']} N; dt={c['dt_s']} s",flush=True)
    print('Headless collection: no live viewer or blocking plot windows. Ctrl+C preserves completed and partial data.',flush=True)
    interrupted=False;push_available=True
    batch_started=time.monotonic(); finished_this_session=0
    try:
        for j,e in enumerate(episodes):
            dest=suite/e['episode_id'];dest.mkdir(exist_ok=True)
            statusfile=dest/'status.json'
            if statusfile.exists():
                status=read_json(statusfile)
                if status['status']=='completed' or (status['status']=='failed' and not args.retry_failed):continue
                # Never overwrite partial/failed records on retry.
                archive=suite/'retry_archive';archive.mkdir(exist_ok=True)
                dest.rename(archive/(e['episode_id']+'_'+str(time.time_ns())));dest.mkdir()
            elif any(dest.iterdir()):
                archive=suite/'retry_archive';archive.mkdir(exist_ok=True)
                dest.rename(archive/(e['episode_id']+'_'+str(time.time_ns())));dest.mkdir()
            write_json(dest/'settings.json',e)
            print(f"[{j+1}/{len(episodes)}] {e['episode_id']}: {e['mass_kg']:g} kg, {e['peak_hand_speed_m_s']:.3g} m/s, start {e['start_deg']:.1f} deg",flush=True)
            with (dest/'console.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(dest)],stdout=log,stderr=subprocess.STDOUT)
                try:
                    proc.wait(timeout=c['episode_timeout_s'])
                except (KeyboardInterrupt,subprocess.TimeoutExpired) as exc:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    if isinstance(exc,KeyboardInterrupt):raise
            if not statusfile.exists() or read_json(statusfile)['status']=='running':
                write_json(statusfile,dict(episode_id=e['episode_id'],task=e['task'],split=e['split'],status='failed',
                                          accepted_for_imitation=False,error='Worker interrupted, crashed, or exceeded time limit; partial CSV preserved.'))
            r=read_json(statusfile); m=r.get('motion_only') or {}
            print(f"  {r['status']}; accepted={r['accepted_for_imitation']}; motion RMSE={m.get('rmse_deg','unavailable')}",flush=True)
            summarize(suite)
            from export_training_data import export
            export(suite, quiet=True)
            finished_this_session+=1
            elapsed=time.monotonic()-batch_started
            remaining=sum(not (suite/item['episode_id']/'status.json').exists() for item in episodes)
            print(f"  Session elapsed: {elapsed/60:.1f} min; remaining estimate: {elapsed/max(1,finished_this_session)*remaining/60:.1f} min (approximate).",flush=True)
            if not args.no_push and push_available:
                try:push_available=publish(dest)
                except Exception as exc:
                    push_available=False;print('Upload failed (collection continues):',exc,flush=True)
                if not push_available:print('Automatic uploads paused for this session; retry saved episodes after authenticating Git.',flush=True)
            elif not args.no_push:
                (dest/'github_push_status.txt').write_text('Queued locally: earlier upload failed. Use RETRY_UPLOADS.bat.\n',encoding='utf-8')
    except KeyboardInterrupt:
        interrupted=True;print('\nInterrupted. Resume with RESUME_COLLECTION.bat; all saved files retained.',flush=True)
    counts=summarize(suite)
    if not interrupted:
        from export_training_data import export
        export(suite)
    batch_summary=suite/'_batch_summary';batch_summary.mkdir(exist_ok=True)
    for name in ['collection_config.json','episode_plan.json','provenance.json','summary.json','summary.csv','counts.json']:
        if (suite/name).exists():shutil.copy2(suite/name,batch_summary/name)
    (batch_summary/'README.txt').write_text('Batch-level summary. Full episode data are sibling folders in the same GitHub batch directory. No NN has been trained.\n',encoding='utf-8')
    if not args.no_push and push_available:
        try:publish(batch_summary)
        except Exception as exc:print('Batch summary upload failed; local files retained:',exc,flush=True)
    print('COUNTS:',json.dumps(counts),flush=True)
    print('RESULTS:',suite/'VIEW_RESULTS.html',flush=True)
    if not args.no_open:webbrowser.open((suite/'VIEW_RESULTS.html').as_uri())
    if any(v['accepted']==0 for v in counts.values()):
        print('WARNING: A task has no accepted demonstrations. Do not train on rejected data as successful demonstrations. Inspect tracking, gains and force limits.',flush=True)
    return 130 if interrupted else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        traceback.print_exc();raise SystemExit(1)
