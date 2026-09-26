from pathlib import Path
import re
import sys

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def apply_patch(target_path: str, patch_path: str) -> None:
    target = Path(target_path)
    patch = Path(patch_path)
    old = target.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = patch.read_text(encoding="utf-8").splitlines(keepends=True)

    out = []
    src = 0
    i = 0
    while i < len(diff):
        m = HUNK_RE.match(diff[i])
        if not m:
            i += 1
            continue

        old_start = int(m.group(1)) - 1
        if old_start < src:
            raise RuntimeError(f"Overlapping hunks in {patch_path}")
        out.extend(old[src:old_start])
        src = old_start
        i += 1

        while i < len(diff) and not diff[i].startswith("@@ "):
            line = diff[i]
            if line.startswith(("--- ", "+++ ")):
                i += 1
                continue
            if line.startswith("\\ No newline"):
                i += 1
                continue
            if not line:
                i += 1
                continue

            op = line[0]
            body = line[1:]
            if op == " ":
                if src >= len(old) or old[src] != body:
                    raise RuntimeError(
                        f"Context mismatch in {target_path} at source line {src + 1}: "
                        f"expected {body!r}, got {old[src] if src < len(old) else '<EOF>'!r}"
                    )
                out.append(old[src])
                src += 1
            elif op == "-":
                if src >= len(old) or old[src] != body:
                    raise RuntimeError(
                        f"Removal mismatch in {target_path} at source line {src + 1}: "
                        f"expected {body!r}, got {old[src] if src < len(old) else '<EOF>'!r}"
                    )
                src += 1
            elif op == "+":
                out.append(body)
            else:
                raise RuntimeError(f"Unsupported diff line: {line!r}")
            i += 1

    out.extend(old[src:])
    target.write_text("".join(out), encoding="utf-8")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: apply_unified_patch.py TARGET PATCH")
    apply_patch(sys.argv[1], sys.argv[2])
