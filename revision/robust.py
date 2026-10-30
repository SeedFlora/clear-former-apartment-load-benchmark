"""
Crash-tolerant runner for GPU jobs on a shared machine.

Each job runs in a fresh worker process (max_tasks_per_child=1), so a CUDA
failure caused by another process filling the GPU cannot corrupt later jobs.
Every finished job is pickled to its own file (written and read only by this
runner, inside results/, so the pickles are trusted); failed jobs are retried
in later rounds.  A retried job starts from the same seed, so its result is the
one an undisturbed run would have produced.
"""
from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _run(fn_name, job, path):
    import jobs as J
    r = getattr(J, fn_name)(job)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(r, f)
    os.replace(path + ".tmp", path)
    return path


def run_jobs(jobs, fn_name, out_dir, workers=4, rounds=8, pause=60):
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, "%04d.pkl" % i) for i in range(len(jobs))]
    t0 = time.time()
    for rnd in range(rounds):
        todo = [i for i, p in enumerate(paths) if not os.path.exists(p)]
        if not todo:
            break
        print(f"  round {rnd + 1}: {len(todo)} jobs pending", flush=True)
        failed = 0
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
            futs = {ex.submit(_run, fn_name, jobs[i], paths[i]): i for i in todo}
            for k, fut in enumerate(as_completed(futs)):
                try:
                    fut.result()
                except Exception as e:                       # noqa: BLE001 - logged and retried
                    failed += 1
                    print(f"    job {futs[fut]} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                if (k + 1) % 20 == 0:
                    print(f"    {k + 1}/{len(todo)} finished, {time.time() - t0:.0f}s", flush=True)
        if failed:
            time.sleep(pause)
    missing = [i for i, p in enumerate(paths) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"{len(missing)} jobs still failing after {rounds} rounds")
    out = []
    for p in paths:
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out
