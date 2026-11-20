"""Small functional tests; test fixtures are not robot experiment results."""
import io,json,tempfile,unittest
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from imitation_lab.raw import Episode,describe
from imitation_lab.convert import clips,split_ids,convert
from imitation_lab.evaluate import action_metrics
from imitation_lab.robot import MotionLimits


def fixture(path,episode_id='fixture',actions=True):
    manifest={'schema':'nexarm-multimodal/1','id':episode_id,'episode_index':0,'fps':5,
              'axes':['J1','J2','J3','J4','J5','gripper'],'units':['degree']*5+['open_fraction'],
              'result':'success','metadata':{'task':'software integration fixture'}}
    rows=[];im=io.BytesIO();Image.new('RGB',(64,64),'gray').save(im,format='JPEG')
    with ZipFile(path,'w') as z:
        for i in range(12):
            row={'episode_index':0,'frame_index':i,'timestamp':i/5,
                 'observation.state':[0,0,0,0,0,.5],'action':[i*.1,0,0,0,0,.5] if actions else None,
                 'observation.images.third':f'media/{i}.jpg','task':'software integration fixture'}
            rows.append(row);z.writestr(f'media/{i}.jpg',im.getvalue())
        z.writestr('manifest.json',json.dumps(manifest));z.writestr('frames.jsonl','\n'.join(map(json.dumps,rows)))
        z.writestr('segments.jsonl','')
    return path


class PipelineTests(unittest.TestCase):
    def test_real_null_actions_not_fabricated(self):
        with tempfile.TemporaryDirectory() as d:
            p=fixture(Path(d)/'e.zip',actions=False)
            with Episode(p) as ep:self.assertEqual(describe(ep)['actions_present'],0)
            with self.assertRaisesRegex(ValueError,'no usable'):convert([p],Path(d)/'out')
            self.assertFalse((Path(d)/'out').exists())

    def test_no_future_chunk_across_missing_actions(self):
        rows=[{'frame_index':i,'observation.state':[0]*6,'action':[0]*6 if i!=11 else None} for i in range(24)]
        self.assertEqual(clips(rows),[list(range(11)),list(range(12,24))])

    def test_episode_splits(self):
        splits=split_ids([str(i) for i in range(50)])
        self.assertEqual(sorted(map(len,splits.values())),[8,8,34])
        self.assertFalse(set(splits['train'])&set(splits['test']))
        self.assertEqual(splits,split_ids([str(i) for i in range(50)]))

    def test_units_metrics(self):
        result=action_metrics([[0,0,0,0,0,0]],[[1,1,1,1,1,1]])
        self.assertEqual(result['joint_rmse_degree'],1);self.assertEqual(result['gripper_open_closed_accuracy'],0)

    def test_motion_rate(self):
        joints=[{'min':-180,'max':180,'zero':0,'sign':1,'ticks':180/np.pi}]*6
        limits=MotionLimits(joints)
        action=limits.apply([100]*6,np.zeros(6))
        self.assertTrue(np.all(action[:5]<=.4+1e-6));self.assertLessEqual(action[-1],.02+1e-6)

if __name__=='__main__':unittest.main()
