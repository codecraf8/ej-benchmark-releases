"""Integrity checks for ej: runtime manifest, release file hashes, and the runtime's no-unpickle guard of the encoder loader.

Ported from the research repository's release/v1/load.py (file_sha256, check_sha, guard_lowbit), with the same behaviour: a
mismatch raises ValueError before anything is read. Loading is pickle-free (ej.safe) and, once loaded, no runtime module
can unpickle (ej.scope.forbid_unpickling). The maintainer-only pickle export (sha256 of the original .pt files, unpickling
them) is not part of the package: it lives in scripts/maintainer_pickle.py (audit M-25)."""
import hashlib
import os

RUNTIME = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_runtime')
MANIFEST = os.path.join(RUNTIME, 'RUNTIME_SHA256')

# Pickle-free release files, by state key prefix: sha256 of each .safetensors file and of each skeleton's JSON bytes
# (after gunzip, so .json and .json.gz give the same value). Add the released state at packaging time.
KNOWN_RELEASES = {
    '14a3e64fb5675f19': {
        'state.safetensors': '6aeb18821c740fbdec3703ed1fba916a75283a890605423875075f6df18fd1f6',
        'state.json': '22e419bcf66f07bc28d1414e2c00f947c4e560b6a264549756825b67b401c68d',
        'encoder/w23.safetensors': '9e996615fb374cd39af6507f2d88c8cd6fa46ef87b09ed946a2605a083c326ed',
        'encoder/w23.json': '66ef3ad38574ee50efad18e723a8135e9709b96dd1d53b42e0b3a717e1f2316f',
    },
    '3b3e66d28fb423f9': {  # v1.0.1: licence fix of the withdrawn v1.0.0 (pool v2b, encoder lowbit-b3b010513f948ceb)
        'state.safetensors': '543e00893159fa3c5128e1c02c9d9070e19c091cfcd01dddcb037b5f5feb3bea',
        'state.json': 'f2dd3a3fad2d0cce525f03bd37ef4e8c5aa1f4d64a5f89c03e9a4f15f0c4bfcf',
        'encoder/w23.safetensors': '99272d36fdcb57423df09ea38b03b2676787dfe738661b4808a592d9c615a3a4',
        'encoder/w23.json': '779151bc2e2a2daf15b0ec495e1dd04bb2aa773932bbfb13a577360cd928d829',
    },
}
# Where each known release's code lives (audit M-14): the research tree whose student*/train_* files and pool reproduce the
# state key (scripts/state_key.py), and the public commit holding the runtime the weights were packaged with.
RELEASE_PROVENANCE = {
    '14a3e64fb5675f19': {
        'status': 'v1.0.0, withdrawn before publication (its training pool contains CC BY-NC data; never public on the Hub)',
        'research_repo': 'codecraf8/rev (private)', 'research_commit': 'f46cf7c', 'research_branch': 'edge-master',
        'pool_sha256': '35b11986cdcdda04d3a8dfb430c321a74096b21218eb647cf33470a0897ef67d',
        'runtime_repo': 'codecraf8/ej-benchmark-releases', 'runtime_commit': '3157bb9',
    },
    '3b3e66d28fb423f9': {
        'status': 'v1.0.1, licence fix of the withdrawn v1.0.0 (same architecture); private Hugging Face revision, tag v1.0.1',
        'research_repo': 'codecraf8/rev (private)', 'research_commit': '963a4fc', 'research_branch': 'edge-master',
        'key_tree': 'f46cf7c edge/ with student_lb.CK_DIR = lowbit-b3b010513f948ceb (scripts/state_key.py --ck-dir)',
        'pool_sha256': 'ce1c1a6fecfd62a90317f6efc4f90fd5f9261becb7081fd00235a6f4d5ee9dbe',
        'runtime_repo': 'codecraf8/ej-benchmark-releases', 'runtime_commit': '2502481',
        'hf_repo': '5ak3t/ej (private)', 'hf_revision': '1038d03e1696fa6452749846b6b2e48135c790ec',
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
