#!/usr/bin/env python3
"""Fetch TROIKA dataset (Zhang 2014 treadmill PPG) — STUB.

The original Zhang 2014 hosting on the corresponding author's SJTU page is
gone; the most recent stable mirror landscape is fragile, and the dataset's
"non-commercial research use" terms make redistributing the archive ourselves
inadvisable.

Until a stable mirror is wired in, the practical path is:

  1. Email the corresponding author (Zhonghua Pi, SJTU) or hunt down a
     well-maintained reproduction repo on GitHub that includes the .mat files.
  2. Drop the 12 .mat files (DATA_01_TYPE01.mat .. DATA_12_TYPE02.mat) into
     data/troika_cache/ (this directory).
  3. Add a NOTICE in data/troika_cache/ pointing at Zhang 2014 + the
     redistribution terms.
  4. src/load_troika.py will then load them.

Phase 3 SQI training can proceed on PPG-DaLiA alone — its `walking`, `stairs`,
`table-soccer`, and `cycling` activity windows already provide the motion-
corrupted regime classical DSP fails on, and the activity labels make the
clean-vs-motion split clean. TROIKA would add held-out test diversity, not
training-set coverage.
"""
import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(1)
