"""ACT training using LeRobot policy, dataset and normalization processors."""
import csv
import json
import random
from pathlib import Path
import subprocess
import sys
import numpy as np
from .raw import write_json


def dataset(root,split,chunk_size=1):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=Path(root);info=json.loads((root/'dataset.json').read_text())
    return LeRobotDataset(repo_id=info['repo_id']+'_'+split,root=root/split,
                          delta_timestamps={'action':[i/info['fps'] for i in range(chunk_size)]},
                          download_videos=False)


def make_policy(cfg,meta):
    from lerobot.configs.types import FeatureType,PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    features={}
    for name,feature in meta.features.items():
        if name=='observation.state':features[name]=PolicyFeature(type=FeatureType.STATE,shape=tuple(feature['shape']))
        elif name.startswith('observation.images.'):
            h,w,c=feature['shape'];features[name]=PolicyFeature(type=FeatureType.VISUAL,shape=(c,h,w))
    config=ACTConfig(input_features=features,output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))},
                     device=cfg['device'],push_to_hub=False,**cfg['policy'])
    return ACTPolicy(config).to(cfg['device'])


def loader(ds,batch_size,shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=0)


def train(cfg):
    root=Path(cfg['dataset'])
    if not (root/'dataset.json').exists():raise ValueError('convert episodes first: '+str(root))
    import torch
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if cfg['device'].startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable; select cpu')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    ds=dataset(root,'train',cfg['policy']['chunk_size'])
    policy=make_policy(cfg,ds.meta)
    pre,post=make_act_pre_post_processors(policy.config,dataset_stats=ds.meta.stats)
    opt=torch.optim.AdamW(policy.get_optim_params(),lr=cfg['policy']['optimizer_lr'],
                         weight_decay=cfg['policy']['optimizer_weight_decay'])
    tr=cfg['training'];out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'resolved_config.json',cfg)
    write_json(out/'dataset.json',json.loads((root/'dataset.json').read_text()))
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    val=dataset(root,'val',cfg['policy']['chunk_size']) if (root/'val').exists() else None
    batches=loader(ds,tr['batch_size'],True);iterator=iter(batches)
    log=(out/'loss.csv').open('w',newline='');writer=csv.DictWriter(log,fieldnames=['step','train_loss','l1','kl','val_l1'])
    writer.writeheader()
    for step in range(1,tr['steps']+1):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(batches);batch=next(iterator)
        policy.train();loss,details=policy(pre(batch));opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),10.);opt.step()
        if step%tr['log_freq']==0 or step==1 or step==tr['steps']:
            val_l1=None
            if val:
                policy.eval();total=0.;count=0
                with torch.no_grad():
                    for j,vbatch in enumerate(loader(val,tr['batch_size'],False)):
                        if j>=tr['validation_batches']:break
                        vb=pre(vbatch);prediction=policy.predict_action_chunk(vb)
                        mask=(~vb['action_is_pad']).unsqueeze(-1).expand_as(prediction)
                        total+=(prediction-vb['action']).abs()[mask].sum().item();count+=mask.sum().item()
                val_l1=total/count
            record={'step':step,'train_loss':loss.item(),'l1':details['l1_loss'],
                    'kl':details.get('kld_loss',0),'val_l1':val_l1}
            writer.writerow(record);log.flush();print(json.dumps(record),flush=True)
        if step%tr['save_freq']==0 or step==tr['steps']:
            path=out/'checkpoints'/str(step)/'pretrained_model'
            policy.save_pretrained(path);pre.save_pretrained(path);post.save_pretrained(path)
            write_json(path/'train_config.json',cfg)
            torch.save({'optimizer':opt.state_dict(),'step':step},path.parent/'training_state.pt')
            last=out/'checkpoints'/'last'
            if last.is_symlink():last.unlink()
            last.symlink_to(str(step),target_is_directory=True)
    log.close()
    plot_losses(out/'loss.csv',out/'loss.png')
    return str(out/'checkpoints/last/pretrained_model')


def plot_losses(csv_path,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(csv_path) as f:rows=list(csv.DictReader(f))
    for field in ('train_loss','val_l1'):
        points=[(int(r['step']),float(r[field])) for r in rows if r[field]]
        if points:plt.plot(*zip(*points),label=field)
    plt.xlabel('Step');plt.ylabel('Loss (normalized)');plt.legend();plt.tight_layout();plt.savefig(output);plt.close()
