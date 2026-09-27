from pathlib import Path
import base64
import io
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"

HUNK_RE = re.compile(r"^@@ -(\\d+)(?:,(\\d+))? \\+(\\d+)(?:,(\\d+))? @@")

def apply_patch(target: Path, patch: Path) -> None:
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
            raise RuntimeError("overlapping patch hunks")
        out.extend(old[src:old_start])
        src = old_start
        i += 1
        while i < len(diff) and not diff[i].startswith("@@ "):
            line = diff[i]
            if line.startswith("\\ No newline"):
                i += 1
                continue
            if not line:
                i += 1
                continue
            op, body = line[0], line[1:]
            if op == " ":
                if src >= len(old) or old[src] != body:
                    raise RuntimeError(f"context mismatch in {target} at line {src + 1}")
                out.append(old[src]); src += 1
            elif op == "-":
                if src >= len(old) or old[src] != body:
                    raise RuntimeError(f"removal mismatch in {target} at line {src + 1}")
                src += 1
            elif op == "+":
                out.append(body)
            else:
                raise RuntimeError(f"unsupported patch line: {line!r}")
            i += 1
    out.extend(old[src:])
    target.write_text("".join(out), encoding="utf-8")

def decode_b64(path: Path) -> bytes:
    return base64.b64decode(path.read_text(encoding="ascii"))

def main() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir()

    bundle = decode_b64(ROOT / "app_bundle.tar.gz.b64")
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tf:
        tf.extractall(RUNTIME)

    patch_bytes = decode_b64(ROOT / "v113_patch.tar.gz.b64")
    patch_dir = ROOT / ".v113_patch"
    if patch_dir.exists():
        shutil.rmtree(patch_dir)
    patch_dir.mkdir()
    with tarfile.open(fileobj=io.BytesIO(patch_bytes), mode="r:gz") as tf:
        tf.extractall(patch_dir)

    apply_patch(RUNTIME / "karaim_morph_engine_v1_1.py", patch_dir / "v113_engine.patch")
    apply_patch(RUNTIME / "app.py", patch_dir / "v113_app.patch")

    # In UI, UNKNOWN means "no dialect restriction", not an empty allowed set.
    app_path = RUNTIME / "app.py"
    app_text = app_path.read_text(encoding="utf-8")
    old = 'requested = [] if dialect == "UNKNOWN" else [dialect]'
    new = 'requested = None if dialect == "UNKNOWN" else [dialect]'
    if old in app_text:
        app_text = app_text.replace(old, new)
    elif new not in app_text:
        raise RuntimeError("Could not locate dialect-selection line in app.py")
    app_path.write_text(app_text, encoding="utf-8")

    shutil.rmtree(patch_dir)

if __name__ == "__main__":
    main()
