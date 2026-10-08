"""ej quickstart: load the weights (a local directory, or a Hugging Face repo id once published) and predict one record.

  python examples/quickstart.py --weights ./ej-weights  # a local weights directory (scripts/hf_layout.py layout)
The Hugging Face weights are not public yet (README "Status"), so --weights is required."""
import argparse
import json

import ej


def main():
    """Load, predict ej.EXAMPLE_RECORD, print each question's distribution."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', required=True, help='local weights directory (or a Hugging Face repo id you can access)')
    ap.add_argument('--revision', help='Hugging Face revision (tag or commit) when --weights is a repo id')
    ap.add_argument('--records', help='optional JSONL file of records; prints one JSON line per record')
    a = ap.parse_args()
    model = ej.load(a.weights, revision=a.revision)
    if a.records:
        with open(a.records) as f:
            records = [json.loads(line) for line in f if line.strip()]
        for r, p in zip(records, model.predict(records)):
            print(json.dumps({'id': r.get('id'), 'probs': p}))
        return
    (pred,) = model.predict([ej.EXAMPLE_RECORD])
    for qid, q in ej.EXAMPLE_RECORD['questions'].items():
        top = max(range(len(pred[qid])), key=pred[qid].__getitem__)
        print(f'{qid} ({q["type"]})')
        for k, (o, p) in enumerate(zip(q['options'], pred[qid])):
            print(f'  {p:.3f}  {o["key"]}{"  <- argmax" if k == top else ""}')


if __name__ == '__main__':
    main()
