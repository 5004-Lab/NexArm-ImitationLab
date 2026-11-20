"""CPU integration test with generated fixtures; not a robotics experiment."""
from pathlib import Path
import json
import yaml
from test_pipeline import fixture
from imitation_lab.convert import convert
from imitation_lab.train import train
from imitation_lab.evaluate import evaluate

import tempfile
Path('outputs').mkdir(exist_ok=True)
root=Path(tempfile.mkdtemp(prefix='smoke-',dir='outputs'))
print('Software integration outputs:',root,flush=True)
paths=[fixture(root/f'{i}.zip',episode_id=f'fixture-{i}') for i in range(3)]
convert(paths,root/'dataset',image_size=(64,64))
with open('configs/act_5hz.yaml') as f:cfg=yaml.safe_load(f)
cfg.update(dataset=str(root/'dataset'),output=str(root/'model'),device='cpu')
cfg['policy'].update(pretrained_backbone_weights=None,dim_model=64,dim_feedforward=128,
                     n_encoder_layers=1,n_decoder_layers=1,n_vae_encoder_layers=1)
cfg['training'].update(steps=2,batch_size=2,log_freq=1,save_freq=2,validation_batches=1)
checkpoint=train(cfg)
print(evaluate(root/'dataset',checkpoint,root/'evaluation'))
