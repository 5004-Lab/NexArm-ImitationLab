import argparse,json
from imitation_lab.convert import convert
p=argparse.ArgumentParser();p.add_argument('episodes',nargs='+');p.add_argument('--output',required=True)
p.add_argument('--cameras',nargs='+',choices=['third','wrist','front'],default=['third'])
p.add_argument('--seed',type=int,default=42);p.add_argument('--repo-id',default='local/nexarm_screwdriver_v1')
a=p.parse_args()
try:result=convert(a.episodes,a.output,repo_id=a.repo_id,cameras=a.cameras,seed=a.seed)
except ValueError as error:p.exit(2,str(error)+'\n')
print(json.dumps(result['splits'],ensure_ascii=False,indent=2))
