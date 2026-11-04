#!/usr/bin/env python3
"""AI Hub 피트니스 자세 이미지 라벨 tar -> 풀린 JSON 폴더.

사용:  python prepare_data.py OUT_DIR  기구.tar  바벨_덤벨.tar  맨몸운동.tar  [문서.tar ...]
각 tar 안의 '*.zip.part0'(= 사실상 완전한 zip) 을 열고, 그 안의 숫자별 zip(01..17)을 다시 풀어
OUT_DIR/<이름>/<번호>/Day*_F/D*.json  구조로 만든다. 문서(.xlsx zip)는 OUT_DIR/docs/ 로 풀린다.
"""
import sys, os, tarfile, zipfile, tempfile, shutil, re

def main():
    out = sys.argv[1]; tars = sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    for t in tars:
        base = os.path.splitext(os.path.basename(t))[0]
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(t) as tf:
                tf.extractall(tmp, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(tmp)
            parts = [os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f.endswith(".part0") or f.endswith(".zip")]
            for p in parts:
                # 분할 압축은 part0 하나뿐이면 그대로 zip. 여러 조각이면 이어붙임(.part0,.part1,...)
                stem = re.sub(r"\.part\d+$", "", p)
                siblings = sorted([q for q in os.listdir(os.path.dirname(p)) if q.startswith(os.path.basename(stem) + ".part")],
                                  key=lambda s: int(re.search(r"part(\d+)$", s).group(1))) if p.endswith(".part0") else []
                z = p
                if len(siblings) > 1:
                    z = stem
                    with open(z, "wb") as w:
                        for s in siblings:
                            with open(os.path.join(os.path.dirname(p), s), "rb") as r: shutil.copyfileobj(r, w)
                if not zipfile.is_zipfile(z):
                    print("skip (not zip):", z); continue
                is_doc = "Document" in p or "문서" in p or p.endswith("명명규칙.zip.part0")
                dest = os.path.join(out, "docs", base) if is_doc else os.path.join(out, base)
                os.makedirs(dest, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    names = zf.namelist()
                    for n in names:
                        if n.endswith(".zip"):
                            m = re.search(r"(\d+)\.zip$", n)
                            sub = os.path.join(dest, m.group(1) if m else os.path.basename(n)[:-4])
                            os.makedirs(sub, exist_ok=True)
                            with zf.open(n) as src, tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmpz:
                                shutil.copyfileobj(src, tmpz); tp = tmpz.name
                            with zipfile.ZipFile(tp) as inner: inner.extractall(sub)
                            os.remove(tp)
                        else:
                            zf.extract(n, dest)
                print(f"extracted {os.path.basename(t)} -> {dest}")

if __name__ == "__main__":
    main()
