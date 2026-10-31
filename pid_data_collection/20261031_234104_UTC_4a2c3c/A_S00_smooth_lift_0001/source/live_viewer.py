"""One persistent OpenSim 3D viewer per collector; compatible drop-in patch.
Episode workers only publish actual states. They never create native viewers.
The viewer process exits when the collector exits. No extra terminal is opened.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    # Retry sharing conflicts from Windows file scanning without losing physics.
    for attempt in range(5):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.01)


def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class EpisodeViewer:
    def __init__(self, osim, model, state, root):
        self.physics = model
        # Each episode worker is a direct child of the same collection process.
        owner = os.getppid()
        self.session = Path(tempfile.gettempdir()) / ('arm26_live_' + str(owner))
        self.session.mkdir(exist_ok=True)
        self.next_frame = 0.
        self.started = time.perf_counter()
        self.names = model.getStateVariableNames()
        self.names = [self.names.get(i) for i in range(self.names.getSize())]
        launch = self.session / 'launched.json'
        ready = self.session / 'ready.json'
        if not launch.exists():
            # Mark BEFORE launch. A failed/closed viewer must not spawn 200 windows.
            atomic_json(launch, {'owner': owner})
            model.printToXML(str(self.session / 'display_model.osim'))
            with (self.session / 'viewer.log').open('a', encoding='utf-8') as log:
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), '--serve',
                     str(self.session), str(owner), str(Path(root).resolve())],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    close_fds=True)
        deadline = time.monotonic() + 30
        while not ready.exists():
            if (self.session / 'error.txt').exists():
                raise RuntimeError((self.session / 'error.txt').read_text())
            if time.monotonic() >= deadline:
                raise RuntimeError('Viewer startup timed out. See ' + str(self.session / 'viewer.log'))
            time.sleep(.1)
        metadata = json.loads(ready.read_text())
        if not process_alive(metadata['pid']):
            raise RuntimeError('Shared viewer has closed; no automatic replacement will be opened.')
        if metadata['state_names'] != self.names:
            raise RuntimeError('Display and simulation state layouts differ.')
        self.update(state, force=True)
        print('SHARED VIEWER: reusing one window for the whole collection.', flush=True)

    def update(self, state, force=False):
        t = float(state.getTime())
        if not force and t + 1e-9 < self.next_frame:
            return
        self.next_frame = t + 1/30
        values = self.physics.getStateVariableValues(state)
        atomic_json(self.session / 'frame.json', {
            'sequence': time.time_ns(), 'worker': os.getpid(), 'time_s': t,
            'values': [values.get(i) for i in range(values.size())]})
        delay = t - (time.perf_counter() - self.started)
        if delay > 0:
            time.sleep(min(delay, .05))

    def close(self):
        # Retain exactly one viewer across episode boundaries and uploads.
        pass


def serve(session, owner, root):
    native = None
    try:
        import opensim as o
        from original_pi_demo import prepare_display
        prepare_display(o, root / 'arm26_exosuit.osim', root / 'Geometry', True)
        model = o.Model(str(session / 'display_model.osim'))
        # Shoulder is fixed in physics, but the shared display must accept each episode's angle.
        for i in range(model.updCoordinateSet().getSize()):
            model.updCoordinateSet().get(i).setDefaultLocked(False)
        model.setUseVisualizer(True)
        state = model.initSystem()
        viewer = model.updVisualizer()
        native = viewer.updSimbodyVisualizer()
        names = model.getStateVariableNames()
        names = [names.get(i) for i in range(names.getSize())]
        viewer.show(state)
        atomic_json(session / 'ready.json', {'pid': os.getpid(), 'state_names': names})
        last = None
        while process_alive(owner):
            try:
                frame = json.loads((session / 'frame.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                time.sleep(.02)
                continue
            if frame['sequence'] != last:
                values = o.Vector(len(frame['values']), 0.)
                for i, value in enumerate(frame['values']):
                    values.set(i, value)
                state.setTime(frame['time_s'])
                model.setStateVariableValues(state, values)
                model.realizeVelocity(state)
                viewer.show(state)
                last = frame['sequence']
            time.sleep(.02)
    except Exception:
        (session / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        if native is not None:
            try:
                native.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--serve':
        serve(Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit('This module is loaded automatically by collect_pid_data.py.')
