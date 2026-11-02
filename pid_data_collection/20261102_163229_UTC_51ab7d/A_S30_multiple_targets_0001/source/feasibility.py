"""Reference force screening using the unchanged OpenSim model and cable cap.
This is a planning estimate; actual closed-loop tracking is validated separately.
"""
import math
import numpy as np
from trajectories import base,configure,reference,FAMILIES
from model_controller import path_moment_arms

def calibrate(cfg,build):
    import opensim as o
    o.Logger.setLevelString('Error')
    results=[]
    for task in cfg['tasks']:
        for shoulder in cfg['shoulder_angles_deg']:
            seed=dict(cfg,task=task['name'],shoulder_deg=shoulder,mass_kg=task['mass_kg']*max(cfg['mass_factor_range']),
                      start_deg=20.,initial_offset_deg=0.,peak_hand_speed_m_s=task['peak_hand_speed_m_s'])
            try:
                model,state,el,cable=build(o,seed);mus=model.getMuscles();solver=o.InverseDynamicsSolver(model)
                probe=o.State(state);el.setSpeedValue(probe,1.);model.realizeVelocity(probe)
                from add_dumbbell import DEFAULT_GRIP
                v=model.getBodySet().get('r_ulna_radius_hand').findStationVelocityInGround(probe,o.Vec3(*DEFAULT_GRIP))
                radius=math.sqrt(sum(v.get(i)**2 for i in range(3)))
                # Equilibrate reference muscle lengths before setting the desired speed.
                def demand(q,v,a):
                    p=o.State(state);el.setValue(p,q,False);el.setSpeedValue(p,0.)
                    model.equilibrateMuscles(p);el.setSpeedValue(p,v);model.realizeDynamics(p)
                    arm=float(path_moment_arms(o,model,p,el,cable,mus)[0])
                    if arm<=cfg['minimum_moment_arm_m']:raise RuntimeError('Insufficient cable moment arm')
                    udot=o.Vector(model.getNumSpeeds(),0.);udot.set(1,a)
                    return float(solver.solve(p,udot).get(1))/arm
                for family in cfg['trajectory_families']:
                    record=dict(task=task['name'],shoulder_deg=shoulder,trajectory=family,
                                requested_peak_hand_speed_m_s=task['peak_hand_speed_m_s'],feasible=False,trials=[])
                    # Static failure cannot be repaired by slowing the trajectory.
                    static=[demand(math.radians(q),0.,0.) for q in np.linspace(19.,80.,32)]
                    record.update(static_min_force_N=min(static),static_max_force_N=max(static))
                    if min(static)<0 or max(static)>cfg['force_cap_N']:
                        record['reason']='Static reference/initial range requires force outside 0..cable cap; no speed or PID gain can fix this with the unchanged actuator.'
                    else:
                        for factor in cfg['speed_screen_factors']:
                            speed=task['peak_hand_speed_m_s']*factor
                            trial=dict(seed,trajectory=family,peak_hand_speed_m_s=speed*max(cfg['speed_factor_range']))
                            configure(trial,radius)
                            forces=[]
                            for t in np.linspace(0,8,81):
                                q,v,a,_=reference(trial['start_hold_s']+t*trial['trajectory_time_scale'],trial)
                                forces.append(demand(q,v,a))
                            record['trials'].append(dict(factor=factor,min_force_N=min(forces),max_force_N=max(forces)))
                            if min(forces)>=0 and max(forces)<=cfg['force_cap_N']*cfg['force_planning_margin']:
                                record.update(feasible=True,selected_peak_hand_speed_m_s=speed,
                                              reason='Reference-force estimate fits the cable bounds; closed-loop check still required.')
                                break
                        if not record['feasible']:record['reason']='No screened speed fits the cable-force planning bounds.'
                    results.append(record)
                    print(f"SCREEN {task['name']} shoulder={shoulder} {family}: "+
                          (f"speed={record['selected_peak_hand_speed_m_s']:.3g} m/s" if record['feasible'] else 'INFEASIBLE'),flush=True)
            except Exception as exc:
                for family in cfg['trajectory_families']:
                    if not any(r['task']==task['name'] and r['shoulder_deg']==shoulder and r['trajectory']==family for r in results):
                        results.append(dict(task=task['name'],shoulder_deg=shoulder,trajectory=family,feasible=False,reason='Screening error: '+str(exc)))
    return results
