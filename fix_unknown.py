from pathlib import Path

p = Path("/app/app.py")
s = p.read_text(encoding="utf-8")
old = 'requested = [] if dialect == "UNKNOWN" else [dialect]'
new = 'requested = None if dialect == "UNKNOWN" else [dialect]'
if old in s:
    s = s.replace(old, new)
elif new not in s:
    raise RuntimeError("Could not locate dialect-selection line in app.py")
p.write_text(s, encoding="utf-8")
print("UNKNOWN dialect handling fixed")
