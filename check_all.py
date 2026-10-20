"""Run every test and print one clear summary.   python check_all.py"""
import glob
import subprocess
import sys
import time

files = sorted(glob.glob("test_*.py"))
failed = []
print(f"Running {len(files)} test files...\n")
for f in files:
    start = time.time()
    run = subprocess.run([sys.executable, "-W", "error::FutureWarning", f], capture_output=True, text=True)
    ok = run.returncode == 0 and "ALL TESTS PASSED" in run.stdout
    print(f"{'PASS' if ok else 'FAIL'}  {f:24s} {time.time() - start:5.1f}s   "
          f"{sum(1 for l in run.stdout.splitlines() if l.startswith('ok')):3d} checks")
    if not ok:
        failed.append(f)
        print("\n".join(run.stdout.splitlines()[-6:]) + "\n" + run.stderr[-600:])
print("\n" + ("EVERYTHING PASSES. Safe to deploy." if not failed else "NOT READY: " + ", ".join(failed)))
sys.exit(1 if failed else 0)
