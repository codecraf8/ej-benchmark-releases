"""Maintainer-only: export the ORIGINAL pickled research files (state-<key16>.pt, lowbit-*/w23.pt) pickle-free.

Not part of the ej package and never reached by ej.load / Model.predict (audit M-25): the package has no unpickling path,
and after ej.load every runtime module's torch.load / pickle.load refuses (ej.scope.forbid_unpickling). Unpickling runs
arbitrary code, so this module refuses unless BOTH hold:
  - the environment variable EJ_MAINTAINER_UNPICKLE=1 is set (an explicit, per-process opt-in), and
  - each file's sha256 equals a known value (KNOWN_STATES / --state-sha for the state, LOWBIT_SHA256 by encoder directory
    for the encoder), checked BEFORE the file is opened by torch.
Used by scripts/hf_layout.py --state-pt / --w23-pt."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'src'))
from ej import integrity, loader, safe  # noqa: E402

OPT_IN = 'EJ_MAINTAINER_UNPICKLE'
# Pickled low-bit encoders (w23.pt) by checkpoint directory, with their verified sha256.
LOWBIT_SHA256 = {
    'lowbit-9c15e5f8f8e3b927': 'd1a0fdf22fabaa3d09efa77c0d5f89fa1e12b6a4f5b57b82bbcd3f72f95abd2f',  # v1.0.0 (pool v2 texts)
    'lowbit-b3b010513f948ceb': 'd386291e794cd69725eb9d8cd89e4b88de8325664af90cd5299fe26dc39c252b',  # v1.0.1 (pool v2b texts)
}
# Pickled state files with their verified sha256 (add a released file at packaging time).
KNOWN_STATES = {
    'state-14a3e64fb5675f19.pt': 'fb4e8bd7c0e9dc31fb6f65a850bbf96fab2fcaca1f932909ca9d8c8e36dbc81f',  # v1.0.0 (withdrawn)
    'state-3b3e66d28fb423f9.pt': '730630f0e130431600539e12aa986068cdb31ae0e03e1a878f58c1aaadd00379',  # v1.0.1
}


class MaintainerOnly(RuntimeError):
    """Raised when the unpickling path is used without the explicit opt-in."""


def require_opt_in():
    """Refuse unless EJ_MAINTAINER_UNPICKLE=1."""
    if os.environ.get(OPT_IN) != '1':
        raise MaintainerOnly(f'unpickling research files is maintainer-only: set {OPT_IN}=1 to run it')


def verify_files(state_path, state_sha256, lowbit):
    """sha256 of the pickled state and of the pickled low-bit encoder (looked up by its directory), before anything is
    unpickled."""
    integrity.check_sha(state_path, state_sha256, 'state file')
    want = LOWBIT_SHA256.get(os.path.basename(os.path.dirname(os.path.abspath(lowbit))))
    if want is None:
        raise MaintainerOnly(f'{lowbit}: encoder directory not in LOWBIT_SHA256; refusing to unpickle it')
    integrity.check_sha(lowbit, want, 'low-bit encoder')


def export_pickles(state_pt, w23_pt, state_sha, tmp):
    """Check the opt-in and both sha256, unpickle, export pickle-free into tmp; returns (state prefix, encoder prefix)."""
    require_opt_in()
    sha = state_sha or KNOWN_STATES.get(os.path.basename(state_pt))
    if not sha:
        raise MaintainerOnly(f'no known sha256 for {state_pt}; pass --state-sha')
    verify_files(state_pt, sha, w23_pt)
    loader.prepare_environment(tmp)  # the pickles name runtime classes by bare module name
    import torch
    st = safe.export(torch.load(state_pt, map_location='cpu', weights_only=False), os.path.join(tmp, 'state'))
    w = safe.export(torch.load(w23_pt, map_location='cpu', weights_only=False), os.path.join(tmp, 'w23'))
    bad = set(st['classes']) - safe.ALLOWED_CLASSES
    if bad or w['classes']:
        raise MaintainerOnly(f'classes outside the allowlist: {sorted(bad) + w["classes"]}')
    return st['json'][:-5], w['json'][:-5]
