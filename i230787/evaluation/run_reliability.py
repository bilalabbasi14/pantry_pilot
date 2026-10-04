"""Custom reliability evaluation over HTTP: Arena stress categories plus multi-turn /chat sequences.
Calls the configured model through a running server, so it may incur provider cost.

    python evaluation/run_reliability.py --url http://127.0.0.1:8000 --delay 8
"""
import argparse, json, time, uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import httpx

HERE = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--delay', type=float, default=5, help='seconds between requests (provider rate limits)')
parser.add_argument('--out', default=str(HERE / 'results'))
args = parser.parse_args()
base = args.url.rstrip('/')
suite = json.loads((HERE / 'reliability_cases.json').read_text(encoding='utf-8'))


def content_problems(text, must=(), must_not=()):
    low = text.lower()
    return [f'missing "{s}"' for s in must if s.lower() not in low] + \
           [f'forbidden "{s}"' for s in must_not if s.lower() in low]


def run_arena(case):
    body = {'task': case['task'], 'external_context': case.get('external_context', []),
            'arena_config': {'max_steps': case.get('max_steps', 6), 'fault': case.get('fault', 'none')}}
    r = httpx.post(base + '/arena/run', json=body, timeout=70)
    r.raise_for_status()
    res = r.json()
    problems = [] if res['status'] in case['expected_status'] else [f"status {res['status']} not in {case['expected_status']}"]
    if case.get('expected_stop_reason') and res['stop_reason'] != case['expected_stop_reason']:
        problems.append(f"stop_reason {res['stop_reason']} != {case['expected_stop_reason']}")
    problems += content_problems(res['final_response'], case.get('must_contain', []), case.get('must_not_contain', []))
    tools = [t['tool'] for t in res['tool_calls']]
    outcomes = [t['outcome'] for t in res['tool_calls']]
    problems += [f'forbidden tool {t} executed' for t in case.get('forbidden_tools', []) if t in tools]
    problems += [f'no tool outcome {o}' for o in case.get('expected_tool_outcomes', []) if o not in outcomes]
    return problems, {'status': res['status'], 'stop_reason': res['stop_reason'], 'steps': res['steps'],
                      'tool_calls': [f"{t['tool']}:{t['outcome']}" for t in res['tool_calls']],
                      'latency_ms': round(res['metrics']['latency_ms']), 'model_calls': res['metrics']['model_calls'],
                      'final_response': res['final_response'][:300]}


def run_chat(case):
    session, problems, turns = f'eval-{uuid.uuid4()}', [], []
    for index, (turn, expected) in enumerate(zip(case['turns'], case['expected_status'])):
        if index:
            time.sleep(args.delay)
        expected = expected if isinstance(expected, list) else [expected]
        if turn == '__RESET__':
            status = httpx.delete(f'{base}/chat/{session}', timeout=20).json()['status']
            turns.append({'turn': 'RESET', 'status': status})
            text = ''
        else:
            r = httpx.post(base + '/chat', json={'session_id': session, 'task': turn}, timeout=70)
            r.raise_for_status()
            res = r.json()
            status, text = res['status'], res['final_response']
            turns.append({'turn': turn, 'status': status, 'stop_reason': res['stop_reason'], 'final_response': text[:300]})
        if status not in expected:
            problems.append(f'turn {index + 1} status {status} not in {expected}')
    problems += content_problems(text, case.get('final_must_contain', []), case.get('final_must_not_contain', []))
    httpx.delete(f'{base}/chat/{session}', timeout=20)
    return problems, {'turns': turns}


records = []
for kind, runner in (('arena', run_arena), ('chat', run_chat)):
    for case in suite[kind]:
        try:
            problems, detail = runner(case)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            problems, detail = [f'{type(exc).__name__}: {exc}'[:200]], {}
        records.append({'id': case['id'], 'category': case['category'], 'passed': not problems, 'problems': problems, **detail})
        print(f"{'PASS' if not problems else 'FAIL'} {case['id']} [{case['category']}] " + '; '.join(problems), flush=True)
        time.sleep(args.delay)

by_category = defaultdict(lambda: {'passed': 0, 'total': 0})
for r in records:
    by_category[r['category']]['total'] += 1
    by_category[r['category']]['passed'] += r['passed']
summary = {'run_at': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'url': args.url,
           'total': len(records), 'passed': sum(r['passed'] for r in records),
           'by_category': dict(by_category), 'cases': records}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / 'reliability_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(f"{summary['passed']}/{summary['total']} passed")
for category, c in summary['by_category'].items():
    print(f"  {category:18} {c['passed']}/{c['total']}")
