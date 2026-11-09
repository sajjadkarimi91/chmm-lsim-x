"""Imported INSIDE spawned initialization workers by tests/test_init.py.

`fit_and_record(job)` runs the production worker entry point
(`chmm_lsim.init._fit_channel_job_in_worker`) on a real prepared job, and records
every BLAS/OpenMP thread pool threadpoolctl can see while the real
`GaussianMixture.fit` is executing. It only wraps `fit` for recording; the pool
factory, initializer, import order and fit are the production ones.
"""

from __future__ import annotations

import os


def fit_and_record(job):
    import chmm_lsim.init as init_module

    record: dict = {"worker_limit_active_before_job": init_module._WORKER_THREAD_LIMIT is not None}
    real_import = init_module._import_gaussian_mixture

    def recording_import():
        GaussianMixture = real_import()

        class RecordingGMM(GaussianMixture):
            def fit(self, X, y=None):
                from threadpoolctl import threadpool_info

                record["pools_during_fit"] = [
                    {"api": p["internal_api"], "lib": os.path.basename(p.get("filepath", "")), "num_threads": int(p["num_threads"])}
                    for p in threadpool_info()
                ]
                return super().fit(X, y)

        return RecordingGMM

    init_module._import_gaussian_mixture = recording_import
    try:
        gmm = init_module._fit_channel_job_in_worker(job)
    finally:
        init_module._import_gaussian_mixture = real_import
    record["pid"] = os.getpid()
    record["env"] = {k: os.environ.get(k) for k in init_module._WORKER_THREAD_ENV}
    return gmm, record
