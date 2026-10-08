"""Integrity checks for ej: runtime manifest, release file hashes, and the guards of the maintainer-only pickle path.

Ported from the research repository's release/v1/load.py (file_sha256, check_sha, verify_files, guard_lowbit, KNOWN_STATES,
LOWBIT_SHA256), with the same behaviour: a mismatch raises ValueError before anything is read. User-facing loading is
pickle-free (ej.safe); the pickle constants and guards below are used only by scripts/hf_layout.py when a maintainer
exports the original .pt files."""
import hashlib
import os

RUNTIME = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_runtime')
MANIFEST = os.path.join(RUNTIME, 'RUNTIME_SHA256')

# Maintainer-only pickle path (scripts/hf_layout.py --state-pt / --w23-pt).
LOWBIT_REL = os.path.join('lowbit-9c15e5f8f8e3b927', 'w23.pt')
LOWBIT_SHA256 = 'd1a0fdf22fabaa3d09efa77c0d5f89fa1e12b6a4f5b57b82bbcd3f72f95abd2f'
# Pickled state files with their verified sha256 (add the released file at packaging time).
KNOWN_STATES = {
    'state-14a3e64fb5675f19.pt': 'fb4e8bd7c0e9dc31fb6f65a850bbf96fab2fcaca1f932909ca9d8c8e36dbc81f',
}

# Pickle-free release files, by state key prefix: sha256 of each .safetensors file and of each skeleton's JSON bytes
# (after gunzip, so .json and .json.gz give the same value). Add the released state at packaging time.
KNOWN_RELEASES = {
    '14a3e64fb5675f19': {
        'state.safetensors': '6aeb18821c740fbdec3703ed1fba916a75283a890605423875075f6df18fd1f6',
        'state.json': '22e419bcf66f07bc28d1414e2c00f947c4e560b6a264549756825b67b401c68d',
        'encoder/w23.safetensors': '9e996615fb374cd39af6507f2d88c8cd6fa46ef87b09ed946a2605a083c326ed',
        'encoder/w23.json': '66ef3ad38574ee50efad18e723a8135e9709b96dd1d53b42e0b3a717e1f2316f',
    },
}


def file_sha256(path):
    """Hex sha256 of a file, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def check_sha(path, expected, what):
    """Raise ValueError unless sha256(path) == expected; returns the digest."""
    got = file_sha256(path)
    if got != expected:
        raise ValueError(f'{what} {path}: sha256 {got} != expected {expected}; refusing to load')
    return got


def read_manifest(path=MANIFEST):
    """{file name: sha256} from a sha256sum-format manifest ('#' lines are comments)."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                sha, name = line.split()
                out[name] = sha
    return out


def runtime_sha256(path=MANIFEST):
    """sha256 of the runtime manifest file itself: one value that pins every runtime module."""
    return file_sha256(path)


def verify_runtime(runtime_dir=RUNTIME, manifest=MANIFEST):
    """Check every runtime module against the manifest, and that no unlisted .py file sits next to them."""
    want = read_manifest(manifest)
    have = {f for f in os.listdir(runtime_dir) if f.endswith('.py')}
    extra, missing = sorted(have - set(want)), sorted(set(want) - have)
    if extra or missing:
        raise ValueError(f'ej runtime {runtime_dir}: unlisted files {extra}, missing files {missing}')
    for name, sha in want.items():
        check_sha(os.path.join(runtime_dir, name), sha, 'runtime module')
    return len(want)


def verify_files(state_path, state_sha256, lowbit):
    """Maintainer pickle path: sha256 of a pickled state file and of the pickled low-bit encoder, before unpickling."""
    check_sha(state_path, state_sha256, 'state file')
    check_sha(lowbit, LOWBIT_SHA256, 'low-bit encoder')


def forbid_lowbit_pickle():
    """After a pickle-free install: make the runtime's lazy encoder loader refuse any cache miss instead of unpickling."""
    import student_lb as LB
    if getattr(LB.checkpoint, '_ej_guarded', False):
        return

    def checkpoint(c, _orig=LB.checkpoint):
        path = os.path.join(os.environ.get('EDGE_CKPT', ''), c['dir'], f"{c['lb']}.pt")
        if path not in LB._CK:
            raise ValueError(f'encoder {path} was not installed from safetensors; refusing to unpickle it')
        return _orig(c)
    checkpoint._ej_guarded = True
    LB.checkpoint = checkpoint  # student_lb.encoder looks the name up in its module globals at call time
