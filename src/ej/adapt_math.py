"""The mathematics of workflow adaptation (ej.adapt): per-(question id, option key) logit offsets on top of the zero-shot output.

For a question q of a workflow with option keys O_q, the adapted prediction is

    p_b(y | x) = softmax_y(log p0(y | x) + b[q, y]),        p0 = the zero-shot distribution of ej

and the offsets b are fitted on the workflow's own records. The model is never fine-tuned.

* Labelled examples only (the OPTION TILT): b ~ N(0, sb^2) per (q, y), maximum a posteriori on the labelled questions
  (L-BFGS, ``fit_tilt``). It learns the workflow's label prior, which a zero-shot model cannot know.
* Unlabelled requests (EXPERIMENTAL): b_q = kappa * D_q + e_q. D_q is the batch-calibration direction (Zhou et al., ICLR 2024,
  arXiv:2309.17249): -log of the mean zero-shot distribution over the workflow's requests, centred, then shrunk toward a target
  for sampling noise (positive-part normal-means shrinkage, ``shrink``; the target is the same direction estimated on the
  workflow's other questions with the same option keys, itself shrunk toward 0). kappa ~ N(kappa0, skappa^2) is the share of the
  batch marginal treated as model bias (unlabelled data alone cannot separate bias from a real label prior, so the prior decides);
  e_q ~ N(0, sb^2) is the option tilt, informed by labelled examples when there are any (``fit_hier``, Newton's method).
  Without labels, b = kappa0 * D.
* The direction needs only running moments of the zero-shot distributions (``Moments``): counts, means and co-moments per
  (question id, option-key set), merged with the parallel update of Chan et al. A workflow can therefore update its correction
  batch after batch without storing requests, and feeding the same requests in any batching gives the same offsets."""
import math

import torch

DT = torch.float64


def merge(a, b):
    """Merge two moment triples (n, mean (k,), co-moment (k, k)) over the same key set (Chan et al.); a may be None."""
    if a is None:
        return b
    n = a[0] + b[0]
    d = b[1] - a[1]
    return n, a[1] + d * (b[0] / n), a[2] + b[2] + torch.outer(d, d) * (a[0] * b[0] / n)


def usable(lp):
    """Boolean mask of rows whose log-probabilities hold no NaN / +inf and at least one finite entry."""
    return ~(torch.isnan(lp) | (lp == math.inf)).any(-1) & torch.isfinite(lp).any(-1)


class Moments:
    """Running moments of zero-shot distributions, per question id and option-key set. Stores no request."""

    def __init__(self):
        """Empty moments ({qid: {sorted key tuple: (n, mean, co-moment)}})."""
        self.stats = {}

    def copy(self):
        """A copy that can be extended without changing this one."""
        out = Moments()
        out.stats = {q: dict(st) for q, st in self.stats.items()}
        return out

    def add(self, rows, lp):
        """Fold rows [(qid, option keys)] with log-probabilities lp (n, K) (-inf beyond a row's options) into the moments.
        Rows that are not ``usable`` are skipped. Returns self."""
        ok = usable(lp) if len(rows) else torch.zeros(0, dtype=torch.bool)
        groups = {}
        for n, (qid, keys) in enumerate(rows):
            if not ok[n]:
                continue
            ks = tuple(sorted(keys))
            p = torch.softmax(lp[n, :len(keys)], -1)
            pos = {k: j for j, k in enumerate(keys)}
            groups.setdefault((qid, ks), []).append(p[[pos[k] for k in ks]])
        for (qid, ks), ps in groups.items():
            X = torch.stack(ps).to(DT)
            mu = X.mean(0)
            C = X - mu
            st = self.stats.setdefault(qid, {})
            st[ks] = merge(st.get(ks), (X.shape[0], mu, C.T @ C))
        return self

    def directions(self, pool=True, shrinkage=True):
        """({(qid, key): D}, [shrinkage weight per qid]); pool: shrink toward the workflow's other qids with the same keys."""
        Dd, lams = {}, []
        for q, st in self.stats.items():
            names = union(st)
            d, tv = bc_direction(st)
            if not shrinkage:
                D, lam = d, 0.0
            else:
                t = None
                peers = [s2 for q2, s2 in self.stats.items() if pool and q2 != q and union(s2) == names]
                if peers:
                    pooled = {}
                    for s2 in peers:
                        for ks, mom in s2.items():
                            pooled[ks] = merge(pooled.get(ks), mom)
                    t = shrink(*bc_direction(pooled))[0]
                D, lam = shrink(d, tv, t)
            Dd.update({(q, k): float(D[j]) for j, k in enumerate(names)})
            lams.append(lam)
        return Dd, lams


def union(st):
    """Sorted option-key union of one question id's moments."""
    return sorted({k for ks in st for k in ks})


def bc_direction(st):
    """(d (K,), trace of its sampling covariance) of one question id: d = -log m centred, m = mean zero-shot probability of each
    option over the rows carrying it; the noise trace (delta method) uses the rows that carry every option (inf if < 2)."""
    names = union(st)
    K, pos = len(names), {k: j for j, k in enumerate(names)}
    s, c = torch.zeros(K, dtype=DT), torch.zeros(K, dtype=DT)
    for ks, (n, mu, _) in st.items():
        ix = [pos[k] for k in ks]
        s[ix] += n * mu
        c[ix] += n
    m = (s / c.clamp(min=1)).clamp(min=1e-12)
    pres = c > 0
    d = torch.zeros(K, dtype=DT)
    lm = -m[pres].log()
    d[pres] = lm - lm.mean()
    full = st.get(tuple(names))
    if full is None or full[0] < 2:
        return d, math.inf
    nf, mu, M2 = full
    mf = mu.clamp(min=1e-12)
    A = (M2 / (nf - 1)) / torch.outer(mf, mf) / nf
    return d, float(A.trace() - A.sum() / K)


def shrink(d, tv, t=None):
    """Positive-part shrinkage of d toward t (default 0): (t + (1 - lam)(d - t), lam), lam = min(1, tv / ||d - t||^2)."""
    t = torch.zeros_like(d) if t is None else t
    ss = float(((d - t) ** 2).sum())
    lam = 1.0 if ss <= 0 or not math.isfinite(tv) else min(1.0, tv / ss)
    return t + (1 - lam) * (d - t), lam


def index(rows, K):
    """(names, idx (n, K)): the sorted (qid, key) names of rows and each entry's name index (len(names) = padding)."""
    names = sorted({(q, k) for q, keys in rows for k in keys})
    pos = {nm: i for i, nm in enumerate(names)}
    idx = torch.full((len(rows), K), len(names), dtype=torch.long)
    for n, (q, keys) in enumerate(rows):
        idx[n, :len(keys)] = torch.tensor([pos[(q, k)] for k in keys])
    return names, idx


def offsets(rows, b, K):
    """(n, K) offsets of rows from b {(qid, key): value} (0 for unknown names)."""
    off = torch.zeros(len(rows), K, dtype=DT)
    for n, (q, keys) in enumerate(rows):
        for j, k in enumerate(keys):
            off[n, j] = b.get((q, k), 0.0)
    return off


def fit_tilt(rows, y, lp, sb=0.5, iters=60):
    """Option tilt: MAP of b ~ N(0, sb^2) per (qid, key) given labelled rows, labels y (option index) and log-probs lp (n, K)."""
    K = lp.shape[1]
    names, idx = index(rows, K)
    valid = torch.isfinite(lp)
    base = lp.masked_fill(~valid, 0.0)
    y = torch.as_tensor(y, dtype=torch.long)
    b = torch.zeros(len(names) + 1, dtype=DT, requires_grad=True)  # last slot: padding (never valid, no prior)
    opt = torch.optim.LBFGS([b], max_iter=iters, tolerance_grad=1e-7, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        z = (base + b[idx]).masked_fill(~valid, -math.inf)
        loss = -torch.log_softmax(z, -1)[torch.arange(len(y)), y].sum() + 0.5 * (b[:-1] ** 2).sum() / sb ** 2
        loss.backward()
        return loss
    opt.step(closure)
    return {nm: float(b[i]) for i, nm in enumerate(names)}


def fit_hier(lp, idx, P, Doff, y, kappa0=0.5, skappa=0.5, sb=0.5, iters=60, gtol=1e-9):
    """MAP of (e (P,), kappa) in logits = lp + e[idx] + kappa * Doff, e ~ N(0, sb^2), kappa ~ N(kappa0, skappa^2) (skappa = 0:
    kappa fixed), on labelled rows (lp (n, K), -inf = invalid). Newton's method with backtracking. Returns (e, kappa)."""
    use_k = skappa > 0
    valid = torch.isfinite(lp)
    A = torch.nn.functional.one_hot(idx, P + 1)[..., :P].to(DT)
    if use_k:
        A = torch.cat([A, Doff.unsqueeze(-1)], -1)
    A = A * valid.unsqueeze(-1)
    B = lp.masked_fill(~valid, 0.0) + (0.0 if use_k else kappa0 * Doff)
    nth = P + int(use_k)
    prec, mu = torch.full((nth,), 1.0 / sb ** 2, dtype=DT), torch.zeros(nth, dtype=DT)
    if use_k:
        prec[-1], mu[-1] = 1.0 / skappa ** 2, kappa0
    Y = torch.nn.functional.one_hot(torch.as_tensor(y, dtype=torch.long), lp.shape[1]).to(DT)

    def parts(th):
        lq = torch.log_softmax((B + A @ th).masked_fill(~valid, -math.inf), -1)
        p = lq.exp()
        f = -float((lq.masked_fill(~valid, 0.0) * Y).sum()) + 0.5 * float((prec * (th - mu) ** 2).sum())
        g = torch.einsum('nkp,nk->p', A, p - Y) + prec * (th - mu)
        W = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
        return f, g, torch.einsum('nkp,nkl,nlq->pq', A, W, A) + torch.diag(prec)
    th = mu.clone()
    f, g, H = parts(th)
    for _ in range(iters):
        if nth == 0 or g.abs().max() < gtol:
            break
        step, t = torch.linalg.solve(H, g), 1.0
        while t > 1e-8:
            f2, g2, H2 = parts(th - t * step)
            if f2 <= f - 1e-4 * t * float(g @ step):
                break
            t *= 0.5
        th, f, g, H = th - t * step, f2, g2, H2
    return (th[:P], float(th[-1])) if use_k else (th, kappa0)
