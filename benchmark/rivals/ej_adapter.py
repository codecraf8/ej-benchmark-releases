"""ej itself, through its public API: ej.load(EJ_WEIGHTS, revision=EJ_REVISION).predict(records).
EJ_WEIGHTS = a Hugging Face repo id or a local weights directory (default 5ak3t/ej); EJ_REVISION pins a Hub revision
(default v1.0.0).
All questions of a record are answered in one pass; probabilities are used as returned (they already sum to 1)."""
import functools
import os


@functools.lru_cache(maxsize=1)
def load():
    """The ej model named by EJ_WEIGHTS / EJ_REVISION (loaded once per process)."""
    import ej
    import torch
    model = ej.load(os.environ.get('EJ_WEIGHTS', '5ak3t/ej'),
                    revision=os.environ.get('EJ_REVISION', 'v1.0.0'))
    torch.set_num_threads(int(os.environ.get('BENCH_THREADS', '1')))  # the runtime sets 2 threads on import
    return model


def predict(records):
    """[{qid: probs}] for records in the ej input format."""
    return load().predict(records)
