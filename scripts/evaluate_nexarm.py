import argparse,json
from imitation_lab.evaluate import evaluate,summarize_trials
from imitation_lab.robot import rollout
p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
o=sub.add_parser('offline');o.add_argument('--dataset',required=True);o.add_argument('--checkpoint',required=True)
o.add_argument('--split',choices=['train','val','test'],default='test');o.add_argument('--output',required=True);o.add_argument('--device',default='cpu')
r=sub.add_parser('rollout');r.add_argument('--checkpoint',required=True);r.add_argument('--url',default='http://127.0.0.1:8765')
r.add_argument('--joints',default='/home/ztl/nexarm-console/src/joints.json');r.add_argument('--output',default='outputs/shadow.jsonl')
r.add_argument('--steps',type=int,default=100);r.add_argument('--device',default='cpu');r.add_argument('--execute',action='store_true');r.add_argument('--stop-file',default='STOP')
t=sub.add_parser('trials');t.add_argument('csv');a=p.parse_args()
if a.mode=='offline':print(json.dumps(evaluate(a.dataset,a.checkpoint,a.output,a.split,a.device),indent=2))
elif a.mode=='trials':print(json.dumps(summarize_trials(a.csv),indent=2))
else:rollout(a.checkpoint,a.url,a.joints,a.output,a.steps,a.device,a.execute,a.stop_file)
