from pathlib import Path
import base64
import io
import re
import tarfile
import urllib.request

APP = Path("/app")
PATCH_URL = "https://raw.githubusercontent.com/Beeeeeeeeeeep/karaim-morphology-web/main/v113_patch.tar.gz.b64"
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def apply_patch(target: Path, diff_text: str) -> None:
    old = target.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = diff_text.splitlines(keepends=True)
    out, src, i = [], 0, 0
    while i < len(diff):
        m = HUNK_RE.match(diff[i])
        if not m:
            i += 1
            continue
        start = int(m.group(1)) - 1
        if start < src:
            raise RuntimeError("overlapping patch hunks")
        out.extend(old[src:start]); src = start; i += 1
        while i < len(diff) and not diff[i].startswith("@@ "):
            line = diff[i]
            if line.startswith("\\ No newline") or not line:
                i += 1; continue
            op, body = line[0], line[1:]
            if op == " ":
                if src >= len(old) or old[src] != body:
                    raise RuntimeError(f"context mismatch in {target.name} at {src+1}")
                out.append(old[src]); src += 1
            elif op == "-":
                if src >= len(old) or old[src] != body:
                    raise RuntimeError(f"removal mismatch in {target.name} at {src+1}")
                src += 1
            elif op == "+":
                out.append(body)
            else:
                raise RuntimeError(f"bad patch line {line!r}")
            i += 1
    out.extend(old[src:])
    target.write_text("".join(out), encoding="utf-8")

def main() -> None:
    payload = urllib.request.urlopen(PATCH_URL, timeout=20).read()
    archive = base64.b64decode(payload)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        engine_patch = tf.extractfile("v113_engine.patch").read().decode("utf-8")
        app_patch = tf.extractfile("v113_app.patch").read().decode("utf-8")
    apply_patch(APP / "karaim_morph_engine_v1_1.py", engine_patch)
    apply_patch(APP / "app.py", app_patch)
    text = (APP / "karaim_morph_engine_v1_1.py").read_text(encoding="utf-8")
    if 'VERSION = "1.1.3-productive-deep-root"' not in text:
        raise RuntimeError("v1.1.3 version marker missing after patch")
    print("v1.1.3 productive deep-root patch applied")

if __name__ == "__main__":
    main()
