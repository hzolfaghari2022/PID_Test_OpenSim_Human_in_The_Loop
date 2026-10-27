"""Figures and a qualified tracking verdict from actual OpenSim CSV outputs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze(suite, rmse_limit=2., peak_limit=5.):
    suite=Path(suite)
    results=json.loads((suite/'summary.json').read_text())
    records=[];datasets=[]
    for result in results:
        dest=suite/result['experiment']
        cfg=json.loads((dest/'settings.json').read_text())
        rec={'experiment':result['experiment'],'status':result['status']}
        if rec['status']!='completed':
            rec['verdict']='FAILED TRIAL: no tracking success claimed';records.append(rec);continue
        with (dest/'PI_data.csv').open() as f:rows=list(csv.DictReader(f))
        data={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
        t=data['time_s'];active=(t>=.5)&(t<.5+sum(cfg['legs']))
        rising=np.zeros(len(t),dtype=bool);falling=rising.copy();start=.5
        for i,d in enumerate(cfg['legs']):
            mask=(t>=start)&(t<start+d)
            if i%2==0:rising|=mask
            else:falling|=mask
            start+=d
        e=data['error_deg']
        rec.update(rmse_deg=float(np.sqrt(np.mean(e[active]**2))),peak_error_deg=float(np.max(abs(e[active]))),
          lifting_rmse_deg=float(np.sqrt(np.mean(e[rising]**2))),lowering_rmse_deg=float(np.sqrt(np.mean(e[falling]**2))),
          upper_saturation_percent=float(100*np.mean(data['tension_N'][active]>=cfg['tmax']-1e-6)),
          lower_clipping_percent=float(100*np.mean(data['torque_raw_Nm'][active]<0)))
        rec['meets_selected_limits']=rec['rmse_deg']<=rmse_limit and rec['peak_error_deg']<=peak_limit
        rec['verdict']='MEETS SELECTED ERROR LIMITS' if rec['meets_selected_limits'] else 'DOES NOT MEET SELECTED ERROR LIMITS'
        if not rec['meets_selected_limits']:
            if rec['upper_saturation_percent']>0 or rec['lower_clipping_percent']>0:
                rec['interpretation']='Error and command clipping coexist. Inspect their timing; this alone does not prove clipping caused all errors.'
            else:rec['interpretation']='No detected clipping during motion. Investigate gains, velocity feedback and model-based compensation before selecting a neural controller.'
        records.append(rec);datasets.append((rec,cfg,data,active,rising,falling))
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(suite/(name+'.'+ext),dpi=190)
        plt.close(fig)
    if datasets:
        labels=[r['experiment'].replace('_','\n') for r,_,_,_,_,_ in datasets];x=np.arange(len(labels))
        fig,ax=plt.subplots(2,2,figsize=(12,8))
        ax[0,0].bar(x,[r['rmse_deg'] for r,*_ in datasets]);ax[0,0].axhline(rmse_limit,color='red',ls='--',label=f'Selected limit: {rmse_limit:g} deg');ax[0,0].set_ylabel('Motion RMSE (deg)');ax[0,0].legend()
        ax[0,1].bar(x,[r['peak_error_deg'] for r,*_ in datasets]);ax[0,1].axhline(peak_limit,color='red',ls='--',label=f'Selected limit: {peak_limit:g} deg');ax[0,1].set_ylabel('Peak absolute error (deg)');ax[0,1].legend()
        ax[1,0].bar(x-.18,[r['lifting_rmse_deg'] for r,*_ in datasets],.36,label='Lifting');ax[1,0].bar(x+.18,[r['lowering_rmse_deg'] for r,*_ in datasets],.36,label='Lowering');ax[1,0].set_ylabel('Phase RMSE (deg)');ax[1,0].legend()
        ax[1,1].bar(x-.18,[r['upper_saturation_percent'] for r,*_ in datasets],.36,label='At upper force cap');ax[1,1].bar(x+.18,[r['lower_clipping_percent'] for r,*_ in datasets],.36,label='Negative torque requested');ax[1,1].set_ylabel('Motion samples (%)');ax[1,1].legend(fontsize=8)
        for a in ax.flat:a.set_xticks(x,labels);a.grid(axis='y',alpha=.2)
        save(fig,'diagnostic_speed_summary')
        fig,axes=plt.subplots(len(datasets),2,figsize=(13,2.9*len(datasets)),squeeze=False)
        for j,(rec,cfg,d,active,up,down) in enumerate(datasets):
            t=d['time_s'];a,b=axes[j]
            a.plot(t,d['reference_deg'],'k--',label='Reference');a.plot(t,d['angle_deg'],label='Simulated');a.set_ylabel(rec['experiment']+'\nangle (deg)');a.legend(fontsize=8)
            b.plot(t,d['torque_raw_Nm'],label='Requested torque');b.plot(t,d['torque_applied_Nm'],label='Applied torque');b.set_ylabel('Torque (N m)');b.legend(fontsize=8)
            clipped=(d['saturated']>.5)&active
            for plot in (a,b):
                plot.fill_between(t,0,1,where=clipped,transform=plot.get_xaxis_transform(),alpha=.12,color='red',label='_nolegend_');plot.set_xlabel('Time (s)');plot.grid(alpha=.2)
        fig.suptitle('Actual tracking and torque limitation | red shading = command clipping',fontsize=12)
        save(fig,'tracking_and_actuation_limits')
        fig,ax=plt.subplots(1,2,figsize=(12,4))
        for rec,cfg,d,active,up,down in datasets:
            phase=(d['time_s']-.5)/sum(cfg['legs']);ax[0].plot(phase[active],d['error_deg'][active],label=rec['experiment']);ax[1].plot(phase[active],d['tension_N'][active],label=rec['experiment'])
        ax[0].set_ylabel('Error (deg)');ax[1].set_ylabel('Cable tension (N)')
        for a in ax:a.set_xlabel('Fraction of complete motion sequence');a.grid(alpha=.2);a.legend(fontsize=8)
        save(fig,'normalized_motion_comparison')
    (suite/'assessment.json').write_text(json.dumps(dict(rmse_limit_deg=rmse_limit,peak_limit_deg=peak_limit,results=records),indent=2))
    lines=['FIXED-GAIN CONTROLLER ASSESSMENT','',f'Selected illustrative limits: motion RMSE <= {rmse_limit:g} deg and peak error <= {peak_limit:g} deg.',
       'These are configurable engineering criteria, not validated clinical limits.','']
    for r in records:
        lines.append(r['experiment']+': '+r['verdict'])
        if 'rmse_deg' in r:lines.append(f"  RMSE {r['rmse_deg']:.3f} deg; peak error {r['peak_error_deg']:.3f} deg; upper cap {r['upper_saturation_percent']:.1f}%; negative requested torque {r['lower_clipping_percent']:.1f}%.")
        if 'interpretation' in r:lines.append('  '+r['interpretation'])
    lines+=['','What these results answer: whether the selected fixed gains meet the selected limits for the tested model, load and speeds.',
      'What they cannot establish: that PID is incapable after fair tuning, or that a neural network is necessary or superior.',
      'The supplied controller is PI when Kv=0. Positive Kv adds actual-velocity damping, not full error-derivative PID.',
      'The single flexor cable cannot actively produce extension torque. A neural network does not remove that physical limitation.',
      'RMSE excludes holds; failed trials remain failures. No automatic gain tuning is performed.']
    (suite/'ANSWER_READ_FIRST.txt').write_text('\n'.join(lines)+'\n')
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path);p.add_argument('--rmse-limit',type=float,default=2.);p.add_argument('--peak-limit',type=float,default=5.)
    a=p.parse_args()
    if a.rmse_limit<=0 or a.peak_limit<=0:p.error('Limits must be positive.')
    analyze(a.suite,a.rmse_limit,a.peak_limit)
    print('Assessment and figures saved in',a.suite)
