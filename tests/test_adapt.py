"""Workflow adaptation (ej.adapt, ej.adapt_math). The first tests use a stand-in model and need no weights; the parity tests at
the end run only when EJ_WEIGHTS_DIR names an ej weights directory."""
import copy
import hashlib
import json
import math
import os

import pytest
import torch

import ej
from ej import adapt_math as AM
from ej.adapt import AdaptedModel, _rows

WEIGHTS = os.environ.get('EJ_WEIGHTS_DIR')
HERE = os.path.dirname(os.path.abspath(__file__))
OPTS = [{'key': k, 'text': f'option {k}'} for k in ('a', 'b', 'c')]


class Fake:
    """Deterministic stand-in for ej.Model.predict: a softmax of hashed scores, biased toward each question's first option."""

    def __init__(self, nan_ids=()):
        self.nan_ids, self.calls = set(nan_ids), 0

    def predict(self, records):
        self.calls += 1
        out = []
        for r in records:
            d = {}
            for qid, q in r['questions'].items():
                h = hashlib.sha256(f'{r["state"]}|{qid}'.encode()).digest()
                z = torch.tensor([h[j] / 64.0 + (1.5 if j == 0 else 0.0) for j in range(len(q['options']))], dtype=torch.float64)
                p = torch.softmax(z, -1).tolist()
                d[qid] = [float('nan')] * len(p) if r.get('id') in self.nan_ids else p
            out.append(d)
        return out


def rec(i, answer=None, n_opts=3, qid='q'):
    opts = [{'key': f'k{j}', 'text': f'option {j}'} for j in range(n_opts)] if n_opts != 3 else OPTS
    r = {'id': f'r{i}', 'state': f'request number {i}', 'questions': {qid: {'type': 'choice', 'instructions': 'Pick one.',
                                                                            'options': opts}}}
    if answer is not None:
        r['answers'] = {qid: answer}
    return r


def maxdiff(a, b):
    return max(abs(x - y) for u, v in zip(a, b) for q in u for x, y in zip(u[q], v[q]))


def test_no_inputs_returns_the_model_itself():
    m = ej.Model(None, {}, None)
    assert m.adapt() is m and m.adapt([], []) is m and m.adapt(examples=None, unlabeled=None) is m


def test_examples_without_answers_change_nothing():
    base, recs = Fake(), [rec(i) for i in range(6)]
    ad = AdaptedModel(base, examples=recs[:3])
    assert ad.offsets == {} and ad.predict(recs[3:]) == base.predict(recs[3:]) and not ad.experimental


def test_option_tilt_learns_the_label_prior():
    base = Fake()
    ex = [rec(i, 'c') for i in range(8)]
    ad = AdaptedModel(base, examples=ex)
    rows, lp = _rows(ex, base.predict(ex))
    assert ad.offsets == AM.fit_tilt(rows, [2] * 8, lp, sb=0.5)
    assert ad.offsets[('q', 'c')] > 0 > ad.offsets[('q', 'a')]
    p0, p1 = base.predict([rec(99)])[0]['q'], ad.predict([rec(99)])[0]['q']
    assert p1[2] > p0[2] and abs(sum(p1) - 1) < 1e-12


def test_gold_format_and_bad_answers():
    r = rec(0)
    r['gold'] = {'q': {'label': 1}}
    assert ej.adapt.answers_of(r) == {'q': 1}
    with pytest.raises(ej.RecordError, match='not one of its option keys'):
        AdaptedModel(Fake(), examples=[rec(0, 'zzz')])


def test_unlabelled_batch_calibration_corner():
    base, recs = Fake(), [rec(i) for i in range(40)]
    ad = AdaptedModel(base, unlabeled=recs, config={'kappa0': 1.0, 'shrinkage': False})
    m = torch.tensor([d['q'] for d in base.predict(recs)], dtype=torch.float64).mean(0)
    bc = -m.log() + m.log().mean()
    assert ad.experimental and max(abs(ad.offsets[('q', k)] - float(bc[j])) for j, k in enumerate('abc')) < 1e-12
    p = torch.tensor([d['q'] for d in ad.predict(recs)], dtype=torch.float64)
    assert (p.mean(0) - 1 / 3).abs().max() < (m - 1 / 3).abs().max()  # the batch marginal moves toward uniform


@pytest.mark.parametrize('with_examples', [False, True])
def test_online_equals_batch(with_examples):
    base, recs = Fake(), [rec(i) for i in range(60)] + [rec(i, qid='other') for i in range(60, 70)]
    ex = [rec(100 + i, 'abc'[i % 3]) for i in range(5)] if with_examples else None
    batch = AdaptedModel(base, examples=ex, unlabeled=recs)
    online = AdaptedModel(base, examples=ex, unlabeled=recs[:7])
    for s in range(7, 50, 9):
        online.observe(recs[s:min(s + 9, 50)])
    calls = base.calls
    online.predict(recs[50:], observe=True)  # predicting a batch folds it in without another model pass
    assert base.calls == calls + 1 and online.n_unlabeled == batch.n_unlabeled == 70
    assert set(online.offsets) == set(batch.offsets)
    assert max(abs(online.offsets[k] - batch.offsets[k]) for k in batch.offsets) < 1e-9


def test_numerical_edge_cases():
    base = Fake(nan_ids={'r3'})
    big = [rec(i, n_opts=17, qid='big') for i in range(5)]
    one = [rec(50, qid='solo')]  # a question id seen in a single record
    same = [{**rec(60 + i), 'state': 'identical request'} for i in range(6)]  # identical predictions
    unl = [rec(i) for i in range(6)] + big + one + same
    for ex in ([rec(90, 'b')], [rec(91, 'k16', n_opts=17, qid='big')]):  # k = 1 labelled example
        ad = AdaptedModel(base, examples=ex, unlabeled=unl)
        assert ad.offsets and all(math.isfinite(v) for v in ad.offsets.values())
        assert len([k for k in ad.offsets if k[0] == 'big']) == 17
        for r, d in zip(unl, ad.predict(unl)):  # r3 has NaN zero-shot output (skipped by the fit; returned as is)
            assert r['id'] == 'r3' or all(abs(sum(p) - 1) < 1e-9 for p in d.values())
    assert AdaptedModel(Fake(), unlabeled=one).offsets[('solo', 'a')] == 0.0  # one row: no evidence of a bias, no offset


def test_records_are_not_modified():
    ex, unl = [rec(i, 'a') for i in range(3)], [rec(i) for i in range(3, 9)]
    keep = copy.deepcopy((ex, unl))
    AdaptedModel(Fake(), examples=ex, unlabeled=unl).predict(unl, observe=True)
    assert (ex, unl) == keep


@pytest.fixture(scope='module')
def model():
    if not WEIGHTS:
        pytest.skip('set EJ_WEIGHTS_DIR to an ej weights directory')
    return ej.load(WEIGHTS)


@pytest.fixture(scope='module')
def reference(model):
    """(records, reference predictions for the loaded weights' state key); skips for weights without a reference."""
    with open(os.path.join(HERE, 'data', 'adapt_reference.json')) as f:
        d = json.load(f)
    key = str(model.config.get('state_key'))[:8]
    if key not in d['reference']:
        pytest.skip(f'no adaptation reference for state {key}')
    return d['records'], d['reference'][key]


def test_identity_without_inputs_on_real_weights(model, reference):
    recs, ref = reference
    assert model.adapt() is model and model.adapt().predict(recs[11:]) == model.predict(recs[11:])
    assert maxdiff(model.predict(recs[11:]), ref['zero_shot']) < 1e-6


@pytest.mark.parametrize('arm', ['examples', 'unlabeled', 'both'])
def test_matches_the_reference_implementation(model, reference, arm):
    recs, ref = reference
    ex = recs[:6] if arm in ('examples', 'both') else None
    unl = recs[6:11] if arm in ('unlabeled', 'both') else None
    got = model.adapt(examples=ex, unlabeled=unl).predict(recs[11:])
    assert maxdiff(got, ref[arm]) <= 1e-6
    assert maxdiff(got, ref['zero_shot']) > 1e-4  # the adaptation does something
