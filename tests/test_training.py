import unittest
from copy import deepcopy

import torch

from visual_goal_world_model import Config, WorldModel
from visual_goal_world_model.training import FrameBank, encode_histories, joint_step, make_optimizer


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.ids = torch.tensor([[-1,-1,0,1], [0,1,2,3], [0,1,2,4], [2,3,4,5], [1,2,3,4]])
        self.bank = FrameBank(torch.randint(256,(8,1,12,12),dtype=torch.uint8))

    def model(self, mode):
        return WorldModel(Config(temporal_mode=mode, image_size=12, patch_size=4,
                                 channels=1, width=16, wm_hidden=24, action_dim=2))

    def test_mask_packing(self):
        bits = torch.tensor([170, 240],dtype=torch.uint8).expand(8,12,2)
        bank = FrameBank(self.bank.images, bits)
        ids = torch.tensor([7,2,2])
        batch = bank.gather(ids)
        torch.testing.assert_close(batch[:,0:1],self.bank.images[ids])
        expected = torch.tensor([1,0,1,0,1,0,1,0,1,1,1,1],dtype=torch.uint8)*255
        torch.testing.assert_close(batch[:,1],expected.expand(3,12,12))

    def test_outputs_and_gradients(self):
        for mode in ("pooled", "patch"):
            with self.subTest(mode=mode):
                encoder = self.model(mode).model.encoder
                other = deepcopy(encoder)
                ids = self.ids
                frames = self.bank.images[ids.clamp_min(0)].float()/255
                expected = encoder(frames, ids >= 0)
                actual = encode_histories(other,self.bank,ids,microbatch=3,checkpoint_patches=True)
                torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
                weights = torch.randn_like(actual)
                (expected*weights).sum().backward()
                (actual*weights).sum().backward()
                for p,q in zip(encoder.parameters(),other.parameters(),strict=True):
                    torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=4e-5)

    def test_joint_objectives_and_ema(self):
        reference = self.model("pooled")
        fast = deepcopy(reference)
        ids = self.ids
        world = (ids, ids.flip(0), ids[:,None].expand(-1,9,-1))
        labeled = (ids, ids[[2,3,2,3,2]],torch.randint(2,(5,2)))
        def update(model, optimized):
            # SGD exposes gradient differences directly in the updated parameters.
            opt = torch.optim.SGD(model.parameters(),lr=0.01)
            return joint_step(model,opt,self.bank,world=world,transitions=(ids,ids.flip(0)),
                              actions=labeled,fast=optimized,microbatch=3,bf16=False)
        normal, quick = update(reference,False), update(fast,True)
        for name in normal:
            torch.testing.assert_close(normal[name],quick[name])
        for p,q in zip(reference.parameters(),fast.parameters(),strict=True):
            torch.testing.assert_close(q,p)
        target_before = next(fast.model.target_encoder.parameters()).clone()
        result = joint_step(fast,make_optimizer(fast),self.bank,world=world,
                            transitions=(ids,ids.flip(0)),actions=None)
        self.assertEqual(result["action"].item(),0)
        self.assertIsNone(fast.head.decoder[-1].weight.grad)
        target = next(fast.model.target_encoder.parameters())
        self.assertFalse(target.requires_grad)
        self.assertFalse(torch.equal(target_before,target))
