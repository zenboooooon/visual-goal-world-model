import tempfile
import unittest
from pathlib import Path

import torch

from visual_goal_world_model import Config, WorldModel


class TestModes(unittest.TestCase):
    def test_modes(self):
        torch.set_num_threads(1)
        for mode in ("pooled", "patch"):
            with self.subTest(mode=mode):
                model = WorldModel(Config(temporal_mode=mode, width=16, wm_hidden=24, image_size=16, channels=1, action_dim=5))
                x = torch.rand(2,4,1,16,16)
                valid = torch.tensor([[False,False,True,True],[True,True,True,True]])
                y = model(x,x,valid,valid)
                self.assertEqual(y.shape,(2,5))
                altered = x.clone()
                altered[~valid] = 100
                torch.testing.assert_close(y,model(altered,x,valid,valid))
                online = model.model.encoder(x,valid)
                target = model.model.target_encoder(x,valid)
                stages = target[:,None].expand(-1,9,-1)
                video = model.world_loss(online,online,stages) + model.transition_loss(online,target)
                video.backward(retain_graph=True)
                self.assertIsNone(model.head.decoder[-1].weight.grad)
                model.action_loss(online[:1],online[:1],torch.zeros(1,5)).backward()
                self.assertIsNotNone(model.head.decoder[-1].weight.grad)
                self.assertFalse(target.requires_grad)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/"model.pt"
                    model.action_names = ["a", "b", "c", "d", "e"]
                    model.save(path)
                    restored = WorldModel.load(path)
                    self.assertEqual(restored.action_names, model.action_names)
                    torch.testing.assert_close(restored(x,x,valid,valid),y)


if __name__ == "__main__":
    unittest.main()
