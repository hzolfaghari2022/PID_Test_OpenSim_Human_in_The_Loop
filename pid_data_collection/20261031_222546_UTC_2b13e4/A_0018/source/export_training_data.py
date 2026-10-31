"""Export retained per-episode arrays without mixing episode-level splits."""
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def export(suite, include_rejected=False, quiet=False):
    suite=Path(suite).resolve()
    plan=json.loads((suite/'episode_plan.json').read_text())
    output=suite/'nn_data';output.mkdir(exist_ok=True)
    manifest=[]
    for task in sorted({e['task'] for e in plan}):
        for split in ['train','validation','test']:
            xs=[];ys=[];ids=[];accept=[];names=None
            for e in plan:
                if e['task']!=task or e['split']!=split:continue
                folder=suite/e['episode_id'];p=folder/'status.json'
                if not p.exists():continue
                r=json.loads(p.read_text());accepted=r.get('accepted_for_imitation',False)
                if not accepted and not include_rejected:continue
                if not (folder/'training_arrays.npz').exists():continue
                with np.load(folder/'training_arrays.npz',allow_pickle=False) as z:
                    if names is not None and not np.array_equal(names,z['feature_names']):raise ValueError('Feature layout mismatch.')
                    names=z['feature_names'];xs.append(z['X_raw']);ys.append(z['y_force_N'])
                    ids.extend([e['episode_id']]*len(z['X_raw']));accept.extend([accepted]*len(z['X_raw']))
            name=f'{task}_{split}'+('_including_rejected' if include_rejected else '_accepted')+'.npz'
            category=output/task/split;category.mkdir(parents=True,exist_ok=True)
            target=category/name
            if xs:
                np.savez_compressed(target,X_raw=np.concatenate(xs),y_force_N=np.concatenate(ys),
                                    episode_id=np.array(ids),accepted=np.array(accept),feature_names=names)
            elif target.exists():
                target.unlink()
            manifest.append(dict(task=task,split=split,file=target.relative_to(output).as_posix() if xs else None,episodes=len(xs),samples=len(ids)))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if not quiet:print(json.dumps(manifest,indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',nargs='?',type=Path);p.add_argument('--include-rejected',action='store_true');a=p.parse_args()
    suite=a.suite or Path((ROOT/'LATEST_DATASET.txt').read_text().strip())
    export(suite,a.include_rejected)
