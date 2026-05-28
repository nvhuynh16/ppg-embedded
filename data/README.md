# Data

By default the project uses a **deterministic synthetic PPG** (72 bpm) so it runs with
zero downloads and no MATLAB. To use a **real PhysioNet PPG** record instead:

```bash
uv add numpy scipy wfdb            # one-time (real-data deps)
uv run python src/reference.py --record bidmc01   # BIDMC PPG & Respiration dataset
```

`wfdb.rdrecord` will fetch the record; the `PLETH`/`PPG` channel is auto-selected, the
band-pass FIR is designed with `scipy.signal.firwin`, and the same fixed-point header is
regenerated. Good public PPG sources: **BIDMC PPG and Respiration**, **PPG-DaLiA**, **MIMIC**.

Ground-truth HR is only known for the synthetic signal; for real records the validation
compares the embedded fixed-point result against the Python float reference.
