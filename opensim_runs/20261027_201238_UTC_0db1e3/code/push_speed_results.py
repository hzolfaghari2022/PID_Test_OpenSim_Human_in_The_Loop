"""Upload a frozen run snapshot using Git installed on the simulation computer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = 'https://github.com/hzolfaghari2022/PID_Test_OpenSim_Human_in_The_Loop.git'
SSH_REPO = 'git@github.com:hzolfaghari2022/PID_Test_OpenSim_Human_in_The_Loop.git'
SUFFIXES = {'.json','.csv','.png','.pdf','.mot','.sto','.osim','.html','.txt','.log'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze(suite, source=None):
    suite = Path(suite).resolve()
    if not (suite/'summary.json').is_file():
        raise ValueError('Select a finished suite folder containing summary.json.')
    snapshot = suite/'upload_snapshot'
    manifest = snapshot/'upload_manifest.json'
    if manifest.exists():
        data=json.loads(manifest.read_text(encoding='utf-8'))
        for name, expected in data['files'].items():
            p=snapshot/name
            if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.is_file() or digest(p)!=expected:
                raise ValueError('Frozen snapshot changed or incomplete: '+name)
        return snapshot
    if snapshot.exists():
        raise ValueError('Incomplete upload_snapshot found. Rename it before retrying to keep its contents.')
    pending=suite/'upload_preparing'
    if pending.exists():raise ValueError('Interrupted upload_preparing folder found; rename it before retrying.')
    pending.mkdir()
    try:
        for p in suite.rglob('*'):
            if pending in p.parents or snapshot in p.parents or not p.is_file():continue
            if p.name in {'github_push_status.txt','github_push_status.json'}:continue
            if p.is_symlink():raise ValueError('Symlink in results: '+p.name)
            if p.suffix.lower() not in SUFFIXES:continue
            target=pending/'results'/p.relative_to(suite);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
        code=pending/'code';code.mkdir()
        for name in ['run_speed_tests.py','analyze_speed_results.py','open_results.py','push_speed_results.py','add_dumbbell.py','requirements.txt']:
            shutil.copyfile(ROOT/name,code/name)
        shutil.copyfile(ROOT/'Python'/'run_arm26_pi_github.py',code/'original_runner.py')
        if source is None:
            settings=next(suite.glob('*/settings.json'))
            cfg=json.loads(settings.read_text());candidate=Path(cfg['model'])
            source = candidate if candidate.is_file() else suite/'loaded_model.osim'
            if not Path(source).is_file() and cfg.get('mass_kg')==0:source=ROOT/'arm26_exosuit.osim'
        if source is None or not Path(source).is_file():
            raise ValueError('Input model unavailable. Supply --source MODEL.osim when retrying an older run.')
        shutil.copyfile(source,pending/'input_model.osim')
        files={p.relative_to(pending).as_posix():digest(p) for p in pending.rglob('*') if p.is_file()}
        if any(p.stat().st_size>=95*1024*1024 for p in pending.rglob('*') if p.is_file()):
            raise ValueError('Snapshot contains a file too large for ordinary Git upload.')
        (pending/'upload_manifest.json').write_text(json.dumps({'suite_id':suite.name,'files':files},indent=2),encoding='utf-8')
        pending.rename(snapshot)
    except Exception:
        shutil.rmtree(pending,ignore_errors=True)
        raise
    return snapshot

def upload(suite, source=None, ssh=False):
    suite=Path(suite).resolve()
    def log(status, message, **extra):
        (suite/'github_push_status.txt').write_text(message+'\n',encoding='utf-8')
        (suite/'github_push_status.json').write_text(json.dumps(dict(status=status,message=message,**extra),indent=2),encoding='utf-8')
        print('GITHUB: '+message,flush=True)
    try:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',suite.name):raise ValueError('Invalid suite folder name.')
        git=shutil.which('git')
        if not git:raise RuntimeError('Install Git for Windows and restart your terminal.')
        spec=importlib.util.spec_from_file_location('git_support',ROOT/'Python'/'run_arm26_pi_github.py')
        support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
        cmd=lambda *args,**kw:support.git_command(git,*args,**kw)
        snapshot=freeze(suite,source)
        remote=SSH_REPO if ssh else REPO
        log('uploading','Uploading results to hzolfaghari2022/PID_Test_OpenSim_Human_in_The_Loop...')
        heads=cmd('ls-remote','--heads',remote)
        refs={line.split()[1] for line in heads.splitlines() if line.strip()}
        if refs and 'refs/heads/main' not in refs:raise RuntimeError('Remote main branch is absent; upload stopped.')
        with tempfile.TemporaryDirectory(prefix='arm26_upload_',ignore_cleanup_errors=True) as temp:
            repo=Path(temp)/'repo'
            args=['clone','--depth','1','--single-branch']
            if refs:args+=['--branch','main']
            cmd(*args,'--',remote,str(repo))
            if not refs:cmd('symbolic-ref','HEAD','refs/heads/main',cwd=repo)
            for key,value in [('user.name','hzolfaghari2022'),('user.email','105059650+hzolfaghari2022@users.noreply.github.com'),('core.autocrlf','false'),('commit.gpgsign','false'),('core.hooksPath',str(Path(temp)/'no-hooks'))]:cmd('config',key,value,cwd=repo)
            relative='opensim_runs/'+suite.name
            target=repo/relative
            if not target.resolve().is_relative_to(repo.resolve()) or target.is_symlink():raise ValueError('Unsafe remote results path.')
            if target.exists():
                old={p.relative_to(target).as_posix():digest(p) for p in target.rglob('*') if p.is_file()}
                new={p.relative_to(snapshot).as_posix():digest(p) for p in snapshot.rglob('*') if p.is_file()}
                if old!=new:raise RuntimeError('A different snapshot already uses this run ID; no files overwritten.')
            else:shutil.copytree(snapshot,target)
            cmd('--literal-pathspecs','add','--',relative,cwd=repo)
            changes=cmd('diff','--cached','--name-only',cwd=repo).splitlines()
            if any(not p.startswith(relative+'/') for p in changes):raise RuntimeError('Unexpected staged paths.')
            if changes:
                cmd('commit','-m','Save OpenSim run '+suite.name,cwd=repo)
                cmd('push','origin','HEAD:refs/heads/main',cwd=repo)
            commit=cmd('rev-parse','HEAD',cwd=repo)
            url='https://github.com/hzolfaghari2022/PID_Test_OpenSim_Human_in_The_Loop/tree/'+commit+'/'+relative
            log('success','SUCCESS: '+url,commit=commit,url=url)
        return True
    except Exception as exc:
        log('failed',str(exc)+' Results remain saved. Retry with RETRY_GITHUB_UPLOAD.bat; do not rerun the simulation.')
        return False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path,nargs='?');p.add_argument('--source',type=Path);p.add_argument('--ssh',action='store_true');a=p.parse_args()
    suite=a.suite
    if suite is None:
        choices=[x for x in (ROOT/'speed_test_results').glob('*') if (x/'summary.json').is_file()]
        if not choices:p.error('No finished suites found.')
        suite=max(choices,key=lambda x:x.name)
    return 0 if upload(suite,a.source,a.ssh) else 1
if __name__=='__main__':raise SystemExit(main())
