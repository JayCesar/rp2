from __future__ import annotations

import sys
from pathlib import Path
from typing import NoReturn

import polars as pl


DIR = Path(__file__).parent
OUT = DIR / "extended_essay-br_preprocessed_for_BLSTM.parquet"
P1 = DIR / "extended_essay-br_preprocessed_for_BLSTM_part1.parquet"
P2 = DIR / "extended_essay-br_preprocessed_for_BLSTM_part2.parquet"


def fuse() -> int:
    if OUT.exists():
        print(f"[skip] {OUT.name} exists")
        return 0
    if not P1.exists() or not P2.exists():
        print(f"[err] missing parts: part1={P1.exists()} part2={P2.exists()}", file=sys.stderr)
        return 2

    lf1 = pl.scan_parquet(P1.as_posix())
    lf2 = pl.scan_parquet(P2.as_posix())
    lf = pl.concat([lf1, lf2], how="vertical")

    try:
        lf.sink_parquet(OUT.as_posix())
    except Exception as e:
        print(f"[warn] sink_parquet failed ({e!r}); fallback to collect/write (uses RAM)", file=sys.stderr)
        df = pl.concat([lf1.collect(streaming=True), lf2.collect(streaming=True)], how="vertical")
        df.write_parquet(OUT.as_posix())

    print(f"[ok] wrote {OUT}")
    return 0


def main() -> NoReturn:
    raise SystemExit(fuse())


if __name__ == "__main__":
    main()
