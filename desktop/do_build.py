import subprocess, sys

# Clean dist
r1 = subprocess.run(
    ['cmd', '/c', 'rmdir /s /q "C:\\Users\\Admin\\wifi vision\\dist\\desktop2"'],
    capture_output=True, text=True
)
print("clean:", r1.returncode, r1.stderr[-100:] if r1.stderr else "ok")

# Build
r2 = subprocess.run(
    ['cmd', '/c', 'build.bat'],
    cwd=r'C:\Users\Admin\wifi vision\desktop',
    capture_output=True, text=True, timeout=300
)
log = r2.stdout + r2.stderr
with open(r'C:\Users\Admin\wifi vision\desktop\build.log', 'w') as f:
    f.write(log)
print("build exit:", r2.returncode)
lines = log.splitlines()
print('\n'.join(lines[-40:]))
