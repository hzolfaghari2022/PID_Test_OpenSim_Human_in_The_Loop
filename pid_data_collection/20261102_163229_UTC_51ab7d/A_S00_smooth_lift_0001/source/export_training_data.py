"""Export accepted episodes, preserving whole-episode splits and category labels."""
import argparse,json,os,ntpath
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def windows_path(path):
    text=ntpath.abspath(str(path))
    if text.startswith('\\\\?\\'):return text
    return '\\\\?\\UNC\\'+text[2:] if text.startswith('\\\\') else '\\\\?\\'+text

def export(suite,include_rejected=False,quiet=False):
    suite=Path(windows_path(suite)) if os.name=='nt' else Path(suite).resolve()
    plan=json.loads((suite/'episode_plan.json').read_text());output=suite/'nn_data';output.mkdir(exist_ok=True)
    manifest=[]
    for task in sorted({e['task'] for e in plan}):
        for split in ['train','validation','test']:
            xs=[];ys=[];ids=[];shoulders=[];families=[];accepted_flags=[];names=None;groups=[]
            for e in plan:
                if e['task']!=task or e['split']!=split:continue
                folder=suite/e['episode_id'];status=folder/'status.json'
                if not status.exists():continue
                accepted=json.loads(status.read_text()).get('accepted_for_imitation',False)
                if not accepted and not include_rejected:continue
                if not (folder/'training_arrays.npz').exists():continue
                with np.load(folder/'training_arrays.npz',allow_pickle=False) as z:
                    if names is not None and not np.array_equal(names,z['feature_names']):raise ValueError('Feature layout mismatch')
                    names=z['feature_names'];xs.append(z['X_raw']);ys.append(z['y_force_N']);n=len(z['X_raw'])
                    ids.extend([e['episode_id']]*n);shoulders.extend([e['shoulder_deg']]*n);families.extend([e['trajectory']]*n);accepted_flags.extend([accepted]*n)
                    groups.append(dict(episode_id=e['episode_id'],trajectory=e['trajectory'],shoulder_deg=e['shoulder_deg'],samples=n))
            folder=output/task/split;folder.mkdir(parents=True,exist_ok=True)
            name='including_rejected.npz' if include_rejected else 'accepted.npz';target=folder/name
            if xs:
                temp=target.with_suffix('.npz.tmp')
                with temp.open('wb') as handle:
                    np.savez_compressed(handle,X_raw=np.concatenate(xs),y_force_N=np.concatenate(ys),feature_names=names,
                        episode_id=np.array(ids),shoulder_deg=np.array(shoulders),trajectory=np.array(families),accepted=np.array(accepted_flags))
                temp.replace(target)
            elif target.exists():target.unlink()
            manifest.append(dict(task=task,split=split,file=target.relative_to(output).as_posix() if xs else None,
                episodes=len(xs),samples=len(ids),categories=groups))
    name='manifest_including_rejected.json' if include_rejected else 'manifest.json'
    temp=output/(name+'.tmp');temp.write_text(json.dumps(manifest,indent=2));temp.replace(output/name)
    if not quiet:print(json.dumps([{k:v for k,v in e.items() if k!='categories'} for e in manifest],indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',nargs='?',type=Path);p.add_argument('--include-rejected',action='store_true');a=p.parse_args()
    export(a.suite or Path((ROOT/'LATEST_DATASET.txt').read_text().strip()),a.include_rejected)
