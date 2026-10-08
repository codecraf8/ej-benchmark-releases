"""Locate, verify and install ej weights (pickle-free), and prepare the runtime environment.

Weights directory layout (built by scripts/hf_layout.py; the Hugging Face repository has the same files):
  config.json                              ej version, runtime sha256, e5 revision, state key, sha256 of every file
  state.safetensors + state.json.gz        fitted state (heads, pool, calibration, TF-IDF, vocabularies)
  encoder/w23.safetensors + encoder/w23.json   low-bit encoder (2-bit weights, 3-bit attention, trimmed vocabulary)
Checks before anything is decoded (verify=True): every file's sha256 against config.json; the files against KNOWN_RELEASES
when the state key is listed there (the trust anchor shipped in this package); config.json's runtime sha256 and e5 revision
against this package; every runtime module against _runtime/RUNTIME_SHA256. Decoding itself never unpickles (ej.safe).

Environment: $EDGE_CKPT is set to the weights directory; $EDGE_CACHE defaults to $XDG_CACHE_HOME/ej (else ~/.cache/ej) and
receives the trimmed tokenizer vocabulary; $HF_HOME defaults to the Hugging Face default location. The base model
intfloat/e5-small-v2 is fetched from the Hub at the pinned revision E5_REVISION on first use."""
import json
import os
import sys

from . import integrity
from .integrity import RUNTIME

E5_MODEL = 'intfloat/e5-small-v2'
E5_REVISION = 'ffb93f3bd4047442299a41ebb6fa998a38507c52'
LAYOUT = {'state': 'state', 'encoder': os.path.join('encoder', 'w23')}  # skeleton/tensor prefixes inside the weights dir
PINNED_CALLS = []  # (class name, model, revision) of every from_pretrained call pin_e5_revision rewrote


def pin_e5_revision():
    """Wrap transformers Auto*.from_pretrained so a call for E5_MODEL without an explicit revision gets E5_REVISION.
    The runtime calls from_pretrained(E5_MODEL) with no revision; the pin keeps that code unchanged. Idempotent."""
    import transformers
    for name in ('AutoModel', 'AutoTokenizer', 'AutoConfig'):
        cls = getattr(transformers, name)
        if getattr(cls.from_pretrained, '_rev_pinned', False):
            continue

        def wrapped(*args, _orig=cls.from_pretrained, _name=name, **kw):
            model = args[0] if args else kw.get('pretrained_model_name_or_path')
            if model == E5_MODEL and kw.get('revision') is None:
                kw['revision'] = E5_REVISION
                PINNED_CALLS.append((_name, model, E5_REVISION))
            return _orig(*args, **kw)
        wrapped._rev_pinned = True
        cls.from_pretrained = staticmethod(wrapped)


def cache_dir():
    """ej's cache directory: $XDG_CACHE_HOME/ej, else ~/.cache/ej."""
    return os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache'), 'ej')


def prepare_environment(weights_dir):
    """Set the variables the runtime reads (EDGE_CKPT, EDGE_CACHE, HF_HOME) and put the runtime on sys.path.
    Must run before the first runtime import: some runtime modules read these variables at import time."""
    os.environ['EDGE_CKPT'] = os.path.abspath(weights_dir)
    os.environ.setdefault('EDGE_CACHE', cache_dir())
    os.makedirs(os.environ['EDGE_CACHE'], exist_ok=True)
    if 'HF_HOME' not in os.environ:  # the runtime would otherwise default to a research-machine path
        xdg = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
        os.environ['HF_HOME'] = os.path.join(xdg, 'huggingface')
    if RUNTIME not in sys.path:
        sys.path.insert(0, RUNTIME)


def check_runtime_imports():
    """Refuse when a bare-name runtime module (e.g. 'student') was imported from somewhere other than ej's runtime."""
    names = integrity.read_manifest()
    for mod in (sys.modules.get(n[:-3]) for n in names):
        f = getattr(mod, '__file__', None)
        if f and os.path.dirname(os.path.realpath(f)) != os.path.realpath(RUNTIME):
            raise ImportError(f'module {mod.__name__!r} was imported from {f}, not from the ej runtime {RUNTIME}; '
                              'rename your module or import ej first')


def resolve(path_or_repo, revision=None):
    """A local weights directory: path_or_repo itself when it is a directory, else a Hugging Face snapshot of that repo id."""
    if os.path.isdir(path_or_repo):
        return os.path.abspath(path_or_repo)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=path_or_repo, revision=revision)


def read_config(weights_dir):
    """config.json of a weights directory (ValueError when missing)."""
    p = os.path.join(weights_dir, 'config.json')
    if not os.path.exists(p):
        raise ValueError(f'{weights_dir}: no config.json; not an ej weights directory (see scripts/hf_layout.py)')
    with open(p) as f:
        return json.load(f)


def _skeleton_sha(weights_dir, prefix):
    import hashlib
    from .safe import read_skeleton_bytes
    return hashlib.sha256(read_skeleton_bytes(os.path.join(weights_dir, prefix))).hexdigest()


def verify_weights(weights_dir, cfg):
    """All release checks (module docstring). Returns the list of warnings (unknown state key)."""
    for rel, sha in sorted(cfg.get('files', {}).items()):
        integrity.check_sha(os.path.join(weights_dir, rel), sha, 'weights file')
    for p in LAYOUT.values():
        rel = p.replace(os.sep, '/')
        if rel + '.safetensors' not in cfg.get('files', {}):
            raise ValueError(f'config.json lists no sha256 for {rel}.safetensors; refusing to load')
    if cfg.get('runtime_sha256') != integrity.runtime_sha256():
        raise ValueError(f'weights were packaged for runtime {cfg.get("runtime_sha256")}, this ej has '
                         f'{integrity.runtime_sha256()}; install the matching ej version')
    if cfg.get('e5_revision') != E5_REVISION:
        raise ValueError(f'weights expect e5 revision {cfg.get("e5_revision")}, this ej pins {E5_REVISION}')
    integrity.verify_runtime()
    known = integrity.KNOWN_RELEASES.get(str(cfg.get('state_key', ''))[:16])
    if known is None:
        return [f'state {str(cfg.get("state_key"))[:16]} is not in ej.integrity.KNOWN_RELEASES: files checked against '
                'config.json only']
    for rel, sha in known.items():
        if rel.endswith('.json'):
            got = _skeleton_sha(weights_dir, rel[:-5].replace('/', os.sep))
            if got != sha:
                raise ValueError(f'{rel}: skeleton sha256 {got} != known {sha}; refusing to load')
        else:
            integrity.check_sha(os.path.join(weights_dir, rel), sha, 'weights file')
    return []


def load_state(weights_dir, verify=True):
    """(fitted state, config) from a weights directory, with the runtime ready to predict. Never unpickles."""
    cfg = read_config(weights_dir)
    for w in (verify_weights(weights_dir, cfg) if verify else ['verification skipped (verify=False)']):
        print(f'ej: warning: {w}', file=sys.stderr)
    prepare_environment(weights_dir)
    check_runtime_imports()
    pin_e5_revision()
    from . import safe
    state = safe.install(os.path.join(weights_dir, LAYOUT['state']), os.path.join(weights_dir, LAYOUT['encoder']),
                         ckpt_dir=weights_dir, key=cfg.get('state_key') if verify else None)
    integrity.forbid_lowbit_pickle()
    import student  # noqa: F401  (runtime entry point; imported here so a bad environment fails at load time)
    check_runtime_imports()
    return state, cfg
