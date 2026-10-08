"""ej quickstart: load the weights (Hugging Face repo id or local directory) and predict one record.

  python examples/quickstart.py                         # weights from Hugging Face (5ak3t/ej @ v1.0.0)
  python examples/quickstart.py --weights ./ej-weights  # a local weights directory (scripts/hf_layout.py layout)"""
import argparse
import json

import ej


def main():
    """Load, predict ej.EXAMPLE_RECORD, print each question's distribution."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', default='5ak3t/ej', help='Hugging Face repo id or local weights directory')
    ap.add_argument('--revision', default='v1.0.0', help='Hugging Face revision (tag or commit)')
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
