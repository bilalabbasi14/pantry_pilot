"""Sends HTTP requests. After implementing a model these may incur provider cost.

A case passes when status and stop_reason match, plus optional must_contain / must_not_contain
checks on final_response and expected_tool_outcomes on tool_calls. Results are saved as JSON.
"""
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import httpx
parser=argparse.ArgumentParser()
parser.add_argument('--url',default='http://127.0.0.1:8000')
parser.add_argument('--cases',default=str(Path(__file__).with_name('public_cases.json')))
parser.add_argument('--out',default=str(Path(__file__).parent/'results'/'public_results.json'))
parser.add_argument('--delay',type=float,default=0,help='seconds between cases (provider rate limits)')
args=parser.parse_args()
cases=json.loads(Path(args.cases).read_text(encoding='utf-8'))


def check(case, result):
    problems=[]
    if result['status']!=case['expected_status']: problems.append(f"status {result['status']} != {case['expected_status']}")
    if result['stop_reason']!=case['expected_stop_reason']: problems.append(f"stop_reason {result['stop_reason']} != {case['expected_stop_reason']}")
    text=result['final_response'].lower()
    problems+=[f'missing "{s}"' for s in case.get('must_contain',[]) if s.lower() not in text]
    problems+=[f'contains forbidden "{s}"' for s in case.get('must_not_contain',[]) if s.lower() in text]
    outcomes=[t['outcome'] for t in result['tool_calls']]
    problems+=[f'no tool outcome {o}' for o in case.get('expected_tool_outcomes',[]) if o not in outcomes]
    return problems


passed,records=0,[]
for index,case in enumerate(cases):
    if index and args.delay: time.sleep(args.delay)
    record={'name':case['name'],'category':case.get('category','')}
    try:
        response=httpx.post(args.url.rstrip('/')+'/arena/run',json=case['request'],timeout=70)
        response.raise_for_status(); result=response.json()
        problems=check(case,result)
        record.update(passed=not problems,problems=problems,status=result['status'],stop_reason=result['stop_reason'],
                      steps=result['steps'],tool_calls=[f"{t['tool']}:{t['outcome']}" for t in result['tool_calls']],
                      latency_ms=round(result['metrics']['latency_ms']),model_calls=result['metrics']['model_calls'],
                      input_tokens=result['metrics']['input_tokens'],output_tokens=result['metrics']['output_tokens'],
                      final_response=result['final_response'][:400])
    except (httpx.HTTPError,KeyError,ValueError) as exc:
        record.update(passed=False,problems=[type(exc).__name__])
    passed+=record['passed']
    records.append(record)
    print(('PASS' if record['passed'] else 'FAIL')+' '+case['name']+('' if record['passed'] else ' -> '+'; '.join(record['problems'])))
summary={'run_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),'url':args.url,'total':len(cases),
         'passed':passed,'failed':len(cases)-passed,'cases':records}
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
Path(args.out).write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(f'{passed}/{len(cases)} checks passed. Results: {args.out}. Public checks are not an Arena score.')
raise SystemExit(0 if passed==len(cases) else 1)
