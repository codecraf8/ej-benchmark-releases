"""Claim lint of the public documents (round-20 audit, docs/audit-r20.md in the research repository). Each test names the
finding it guards; it fails when a withdrawn or unsupported claim comes back, or when a required correction is missing."""
import json
import os
import re

import pytest

import ej

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ['README.md', 'docs/MODEL_CARD.md', 'docs/DATA_CARD.md', 'NOTICE', 'CHANGELOG.md', 'benchmark/README.md',
        'benchmark/METHOD.md', 'benchmark/results/run2/RUN2.md',
        'benchmark/results/run3/RUN3.md']


def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def flat(rel):
    """The document with line breaks collapsed (sentences wrap across lines)."""
    return re.sub(r'\s+', ' ', read(rel))


def test_m10_no_public_quickstart_that_fails_and_versions_agree():
    readme = read('README.md')
    assert 'not yet public' in readme and 'pending' in readme
    assert not re.search(r"ej\.load\(\s*['\"]5ak3t/ej", readme), 'README tells anonymous users to load a private repo'
    assert "default='5ak3t/ej'" not in read('examples/quickstart.py')
    version = re.search(r'^version = "([^"]+)"', read('pyproject.toml'), re.M).group(1)
    assert version == ej.__version__ and f'version = {{{version}}}' in readme


@pytest.mark.parametrize('rel', DOCS)
def test_m7_m9_no_withdrawn_rankings(rel):
    text = flat(rel).lower()
    for claim in ('smallest', 'fastest', 'lowest mean ece', 'best-calibrated', 'best calibrated', '1 cpu thread'):
        hits = [m.start() for m in re.finditer(re.escape(claim), text)]
        for h in hits:  # allowed only where the document says the claim was withdrawn
            assert 'withdrawn' in text[max(0, h - 400):h + 400], (rel, claim)


def test_m1_counterfactual_licence_and_withdrawal():
    notice = flat('NOTICE')
    assert 'CC BY-NC 4.0' in notice and 'https://creativecommons.org/licenses/by-nc/4.0/' in notice
    for rel in ('README.md', 'docs/MODEL_CARD.md', 'docs/DATA_CARD.md'):
        t = flat(rel)
        assert 'withdrawn' in t and 'CC BY-NC 4.0' in t, rel
        assert 'Amazon counterfactual (CC BY 4.0)' not in t, rel


def test_m31_notice_names_creator_link_licence_uri_and_modified_per_source():
    notice = read('NOTICE')
    block = notice[notice.index('Training data (pool v2b)'):notice.index('Withdrawn v1.0.0 weights')]
    entries = [e for e in re.split(r'\n- ', block)[1:] if not e.startswith('In-house')]
    assert len(entries) == 4, entries
    for e in entries:
        assert 'Creator:' in e and 'Link: https://' in e and re.search(r'Licence: .*https://', e, re.S) and 'Modified:' in e, e
    for q in ('Q1', 'Q2', 'Q3', 'Q4'):
        assert re.search(rf'^{q}\s', notice, re.M), q


def test_m2_m12_zs_td_statement_and_selection_line():
    card = flat('docs/MODEL_CARD.md')
    m2 = ('zs_td is ONE held-out workflow (security_incidents; dev 300 and final 100 records of the same workflow). v1.0.0 '
          'was kept on its dev accuracy (A-025) after ~55 logged comparisons on it, so final zs_td estimates accuracy on '
          'further records of a workflow the model was selected on: it is neither unbiased nor unseen-workflow evidence.')
    assert m2 in card
    assert 'Unseen-workflow evidence = zs_wide final macro_real.' in card
    for rel in ('README.md', 'docs/MODEL_CARD.md'):
        assert 'dev numbers are selected (55+ comparisons; every run logged from round 21)' in flat(rel), rel
    for rel in DOCS:
        t = flat(rel).lower()
        assert 'never used for model selection' not in t and 'the unbiased read' not in t, rel


def test_m8_m11_m18_adaptation_table():
    readme = flat('README.md')
    assert '111 public-source + 43 GLM-synthetic' in readme and '154 unseen workflows (public sources)' not in readme
    assert '+.010 [−.002, +.023]: covers 0' in readme
    assert 'count prior' in readme and '+.001 [−.007, +.009]' in readme
    assert 'record-cluster bootstrap' in readme


def test_m17_m22_calibration_and_ca_scope():
    card, readme = flat('docs/MODEL_CARD.md'), flat('README.md')
    assert 'this suite only' in card and 'this suite only' in readme
    assert '**calibrated**' in card and '**not calibrated** on those suites' in card


def test_m24_one_statement_on_transductive_use():
    statement = ('Zero-shot `predict` is record-independent: each record\'s output depends only on that record')
    for rel in ('README.md', 'docs/MODEL_CARD.md'):
        t = flat(rel)
        assert statement in t and 'opt-in, per-workflow transductive mode' in t, rel
        assert 'Transductive use that pools option statistics across records to improve accuracy' not in t, rel


def test_m27_m29_glossary_and_overlap():
    readme, card = flat('README.md'), flat('docs/MODEL_CARD.md')
    assert 'never-seen label space' not in card and 'about 22% of its intent names' in card
    assert '**k (few-shot)**: the number of labelled *records*' in readme
    assert '**tickets_ood**' in readme and 'writing-style shift' in readme
    for rel in ('README.md', 'docs/MODEL_CARD.md'):
        for line in read(rel).splitlines():  # a number next to "accuracy" must say which accuracy
            if 'fit-to-fit SD' in line:  # the zs_td statement: one workflow, where micro and macro coincide
                continue
            if re.search(r'\baccuracy\b', line) and re.search(r'(?<![\w.])0?\.\d{2,}', line):
                assert re.search(r'micro|macro', line), (rel, line)


def test_m5_sizes_name_their_measure():
    for rel in ('README.md', 'docs/MODEL_CARD.md'):
        t = flat(rel)
        assert 'counted' in t and 'on disk' in t and 'download' in t and 'resident' in t, rel
        assert '11.2 MB counted' not in t, rel


def test_m6_m9_latency_numbers_come_from_measured_one_thread_summaries():
    d = os.path.join(ROOT, 'benchmark', 'results', 'latency-r21')
    summaries = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith('.summary.json')]
    assert summaries
    readme = read('README.md')
    for s in summaries:
        assert s['threads'] == 1 and s['threads_measured'] == [1] and s['threads_ok'] and s['rival'] == 'ej'
        assert f"{s['ms_per_record']:,.1f}" in readme, (s['suite'], s['cache'], s['ms_per_record'])
    assert 'not comparable' in readme and 'not comparable' in flat('benchmark/results/run2/RUN2.md')


def test_m30_suite_checksums_and_builders():
    sums = read('benchmark/suites/SHA256SUMS')
    for suite in ('td', 'zs_td', 'zs_massive'):
        for split in ('dev', 'final'):
            assert re.search(rf'^[0-9a-f]{{64}}  {split}/{suite}\.jsonl$', sums, re.M), (split, suite)
    for f in ('build_public.py', 'sources.py', 'leakfree.py'):
        compile(read(f'benchmark/suites/{f}'), f, 'exec')


def test_m37_ci_runs_tests_without_deploying():
    wf = read('.github/workflows/tests.yml')
    on = wf[wf.index('\non:'):wf.index('\npermissions:')]
    assert 'pull_request' in on and 'workflow_dispatch' in on and 'push' not in on and 'schedule' not in on
    assert 'contents: read' in wf and 'pytest' in wf
    for word in ('deploy', 'secrets.', 'upload', 'publish', 'twine', 'huggingface-cli'):
        assert word not in wf.replace('no deployment', ''), word


def test_release_docs_have_no_unfilled_placeholders():
    for rel in DOCS:
        assert not re.findall(r'\{\{[^}]*\}\}|@@[A-Z0-9_]+@@', read(rel)), rel


def test_v101_is_the_licence_fix_and_no_v110_release_is_described():
    for rel in ('README.md', 'docs/MODEL_CARD.md', 'CHANGELOG.md'):
        t = flat(rel)
        assert 'licence fix of the withdrawn v1.0.0; same architecture' in t, rel
        assert 'v1.1.0' not in t and 'ej 1.1.0' not in t, rel
    assert 'macro_real' in flat('README.md') and '.419 [.379, .458]' in flat('README.md')
