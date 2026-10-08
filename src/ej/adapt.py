"""Adapting ej to one workflow: ``model.adapt(examples=None, unlabeled=None)`` returns an ``AdaptedModel``.

ej is trained to answer questions it has never seen, so it cannot know how often each answer occurs in your workflow. Adaptation
adds a per-(question id, option key) offset to the zero-shot log-probabilities (ej.adapt_math), fitted on that workflow's records:

* ``examples``: records of the workflow with their correct answers, as ``'answers': {qid: option key}`` (or the benchmark format
  ``'gold': {qid: {'label': option index}}``). The labelled questions fit the option tilt. A few labelled records are enough.
* ``unlabeled`` (EXPERIMENTAL): records of the workflow without answers (for example yesterday's requests). They move the
  offsets toward a batch-calibrated output; with examples as well, the labels decide how much of that correction to keep.
  ``AdaptedModel.observe(batch)`` and ``predict(batch, observe=True)`` fold further unlabelled batches in as they arrive; only
  running moments are kept, never the requests, and any batching of the same requests gives the same offsets.

Offsets are matched by question id and option key, so use one AdaptedModel per workflow. Without examples and unlabelled records
``model.adapt()`` returns the model itself (identical predictions). Cost: one zero-shot pass over the given records, plus a fit
of a few milliseconds; predicting with an adapted model costs the same as with the base model."""
import torch

from . import adapt_math as AM
from .records import RecordError, validate_records

DEFAULTS = {'sb': 0.5, 'kappa0': 0.5, 'skappa': 0.5, 'pool': True, 'shrinkage': True}
"""sb: prior scale of the option tilt; kappa0 / skappa: prior mean / scale of the bias share of unlabelled corrections; pool and
shrinkage: how the unlabelled direction is denoised (ej.adapt_math). Chosen on development data the model was trained on."""


def answers_of(record):
    """{qid: option index} of the answered questions of a record ('answers' by option key, or 'gold' by option index)."""
    qs, out = record['questions'], {}
    for qid, key in (record.get('answers') or {}).items():
        keys = [o['key'] for o in qs.get(qid, {}).get('options', [])]
        if key not in keys:
            raise RecordError(f'record {record.get("id")!r}: answer {key!r} of {qid!r} is not one of its option keys {keys}')
        out[qid] = keys.index(key)
    for qid, g in (record.get('gold') or {}).items():
        if qid not in out and isinstance(g, dict) and isinstance(g.get('label'), int) and qid in qs:
            if not 0 <= g['label'] < len(qs[qid]['options']):
                raise RecordError(f'record {record.get("id")!r}: gold label of {qid!r} is out of range')
            out[qid] = g['label']
    return out


def _rows(records, preds):
    """([(qid, option keys)], log-probabilities (n, K) float64 with -inf padding) in record / question order."""
    rows = [(qid, [o['key'] for o in q['options']]) for r in records for qid, q in r['questions'].items()]
    K = max((len(k) for _, k in rows), default=1)
    lp = torch.full((len(rows), K), -float('inf'), dtype=AM.DT)
    n = 0
    for r, d in zip(records, preds):
        for qid in r['questions']:
            p = torch.tensor(d[qid], dtype=AM.DT)
            lp[n, :len(p)] = p.clamp(min=1e-300).log()
            n += 1
    return rows, lp


class AdaptedModel:
    """An ej model adapted to one workflow. Create it with ``Model.adapt``; ``predict`` as ``Model.predict``."""

    def __init__(self, base, examples=None, unlabeled=None, config=None):
        """Adapt `base` (an ej.Model, or any object with the same predict) with labelled `examples` and/or `unlabeled` records."""
        self.base, self.config = base, {**DEFAULTS, **(config or {})}
        self.moments, self.n_unlabeled, self.offsets = AM.Moments(), 0, {}
        self._inputs = self._labelled = None
        examples = validate_records(examples or [])
        if examples:
            rows, lp = _rows(examples, base.predict(examples))
            ans = [answers_of(r) for r in examples]
            lab = self._label_index(examples, ans)
            self._inputs = (rows, lp)
            if lab:
                ix = torch.tensor([i for i, _ in lab])
                keep = AM.usable(lp[ix])
                ix = ix[keep]
                self._labelled = ([rows[i] for i in ix.tolist()], [y for (_, y), k in zip(lab, keep.tolist()) if k], lp[ix])
        if unlabeled:
            self._fold(validate_records(unlabeled), None)
        self._refit()

    @staticmethod
    def _label_index(examples, ans):
        """[(row index, option index)] of the answered questions."""
        out, n = [], 0
        for r, a in zip(examples, ans):
            for qid in r['questions']:
                if qid in a:
                    out.append((n, a[qid]))
                n += 1
        return out

    @property
    def experimental(self):
        """True when unlabelled records shape the offsets (the experimental part of adaptation)."""
        return self.n_unlabeled > 0

    def _fold(self, records, preds):
        """Add zero-shot predictions of unlabelled records to the moments (computed here when preds is None)."""
        if preds is None:
            preds = self.base.predict([{k: v for k, v in r.items() if k not in ('answers', 'gold')} for r in records])
        self.moments.add(*_rows(records, preds))
        self.n_unlabeled += len(records)

    def _refit(self):
        """Recompute the offsets from the labelled examples and the moments of the unlabelled records."""
        c = self.config
        if not self.n_unlabeled:  # labelled examples only: the option tilt
            self.offsets = AM.fit_tilt(*self._labelled, sb=c['sb']) if self._labelled else {}
            return
        mom = self.moments.copy()
        if self._inputs:  # the examples' inputs also inform the direction (their answers never do)
            mom.add(*self._inputs)
        Dd, _ = mom.directions(pool=c['pool'], shrinkage=c['shrinkage'])
        b, kappa = {}, c['kappa0']
        if self._labelled:
            rows, y, lp = self._labelled
            names, idx = AM.index(rows, lp.shape[1])
            e, kappa = AM.fit_hier(lp, idx, len(names), AM.offsets(rows, Dd, lp.shape[1]), y, c['kappa0'], c['skappa'], c['sb'])
            b = {nm: float(e[i]) for i, nm in enumerate(names)}
        for k, v in Dd.items():
            b[k] = b.get(k, 0.0) + kappa * v
        self.offsets = {k: v if v == v and abs(v) != float('inf') else 0.0 for k, v in b.items()}

    def observe(self, records):
        """Fold a batch of unlabelled records of the workflow into the adaptation (EXPERIMENTAL) and refit. Returns self."""
        records = validate_records(records)
        if records:
            self._fold(records, None)
            self._refit()
        return self

    def predict(self, records, observe=False):
        """[{qid: [p_1, ..., p_K]}] per record, adapted. observe=True also folds these records in as unlabelled requests
        (EXPERIMENTAL; their zero-shot pass is reused, so this costs no extra model pass) after predicting them."""
        records = validate_records(records)
        if not records:
            return []
        preds = self.base.predict(records)
        out = preds if not self.offsets else self._apply(records, preds)
        if observe:
            self._fold(records, preds)
            self._refit()
        return out

    def _apply(self, records, preds):
        rows, lp = _rows(records, preds)
        p = torch.softmax(lp + AM.offsets(rows, self.offsets, lp.shape[1]).masked_fill(~torch.isfinite(lp), 0.0), -1)
        out, n = [], 0
        for r in records:
            d = {}
            for qid, q in r['questions'].items():
                row = p[n, :len(q['options'])].numpy().astype(float)
                d[qid] = [float(x) for x in row / row.sum()]
                n += 1
            out.append(d)
        return out

    def __repr__(self):
        return (f'ej.AdaptedModel(base={self.base!r}, labelled={len(self._labelled[1]) if self._labelled else 0}, '
                f'unlabeled={self.n_unlabeled}, offsets={len(self.offsets)})')
