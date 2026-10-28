"""Double-click START_LIVE_DUMBBELL.bat, or run this script in lstm-py312."""
from pathlib import Path
import subprocess
import sys
if __name__ == '__main__':
    command = [sys.executable, str(Path(__file__).with_name('run_speed_tests.py')), '--only', 'slow', '--live']
    raise SystemExit(subprocess.call(command + sys.argv[1:]))
