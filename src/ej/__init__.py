"""ej: calibrated typed decisions in one pass, on device.

Give it a `state` (text, or a JSON object as text) and typed questions (choice / noul (yes-no) / score); it returns one
probability distribution per question, without generating tokens.

    import ej
    model = ej.load('5ak3t/ej', revision='v1.0.0')   # or a local weights directory
    (probs,) = model.predict([ej.EXAMPLE_RECORD])          # {qid: [p for each option, in option order]}

The prediction code is the research runtime in ej/_runtime (code unchanged, comments cleaned), imported by bare module name
from a sys.path entry that ej adds on load; do not shadow those names with modules of your own."""
from .records import EXAMPLE_RECORD, NOUL_OPTIONS, RecordError, validate_record, validate_records

__version__ = '1.0.0'
__all__ = ['load', 'Model', 'EXAMPLE_RECORD', 'NOUL_OPTIONS', 'RecordError', 'validate_record', 'validate_records',
           '__version__']


class Model:
    """A loaded ej model. Create it with ej.load(); call predict(records)."""

    def __init__(self, state, config, weights_dir):
        """Wrap a fitted state (from ej.loader.load_state) with its config.json and weights directory."""
        self._state, self.config, self.weights_dir = state, config, weights_dir

    def predict(self, records):
        """[{qid: [p_1, ..., p_K]}] per record, each list in the order of that question's options and summing to 1.
        Records are validated first (ej.records); several records are encoded in one batch."""
        import student  # the runtime is on sys.path once ej.load() has run
        records = validate_records(records)
        if not records:
            return []
        out = student.predict(self._state, records)
        return [{qid: [float(p) for p in ps] for qid, ps in d.items()} for d in out]

    def __repr__(self):
        return f"ej.Model(state={str(self.config.get('state_key'))[:16]}, weights_dir={self.weights_dir!r})"


def load(path_or_hf_repo, revision=None, verify=True):
    """Load ej weights from a local directory or a Hugging Face repo id (pin `revision` to a commit for reproducibility).
    verify=True checks every file's sha256 and the runtime before decoding (ej.loader); loading never unpickles."""
    from . import loader
    weights_dir = loader.resolve(path_or_hf_repo, revision)
    state, cfg = loader.load_state(weights_dir, verify=verify)
    return Model(state, cfg, weights_dir)
