import argparse,yaml
from imitation_lab.train import train
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/act_5hz.yaml')
p.add_argument('--device');p.add_argument('--steps',type=int);p.add_argument('--output');a=p.parse_args()
with open(a.config) as f:cfg=yaml.safe_load(f)
if a.device:cfg['device']=a.device
if a.steps:cfg['training']['steps']=a.steps
if a.output:cfg['output']=a.output
try:print(train(cfg))
except ValueError as error:p.exit(2,str(error)+'\n')
