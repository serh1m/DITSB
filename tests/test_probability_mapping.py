import torch
import unittest
from ditsb.discrete_flow import CategoricalFlowMatcher

class TestCategoricalFlowMatcher(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 10
        self.matcher = CategoricalFlowMatcher(vocab_size=self.vocab_size)
    
    def test_probability_mapping(self):
        # Batch size 4, Seq len 5, Vocab size 10
        batch_size, seq_len = 4, 5
        x1_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        x1_onehot = torch.nn.functional.one_hot(x1_tokens, num_classes=self.vocab_size).float()
        
        # Test sample_pt probabilities sum to 1
        t = torch.rand(batch_size, seq_len, 1)
        pt = self.matcher.sample_pt(x1_onehot, t)
        
        sums = pt.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)), "Probabilities at time t do not sum to 1")
        
        # Test start uniform
        t_zero = torch.zeros(batch_size, seq_len, 1)
        pt_zero = self.matcher.sample_pt(x1_onehot, t_zero)
        uniform_expected = torch.ones_like(x1_onehot) / self.vocab_size
        self.assertTrue(torch.allclose(pt_zero, uniform_expected), "t=0 is not a uniform Dirichlet prior")
        
        # Test end dirac
        t_one = torch.ones(batch_size, seq_len, 1)
        pt_one = self.matcher.sample_pt(x1_onehot, t_one)
        self.assertTrue(torch.allclose(pt_one, x1_onehot), "t=1 is not the target one-hot")
    
    def test_euler_step_discrete(self):
        batch_size, seq_len = 4, 5
        probs = torch.ones(batch_size, seq_len, self.vocab_size) / self.vocab_size
        logits_theta = torch.randn(batch_size, seq_len, self.vocab_size)
        
        dt = 0.1
        probs_next = self.matcher.euler_step_discrete(probs, logits_theta, dt)
        
        # Verify it remains strictly valid probability (sums to 1)
        sums = probs_next.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "Euler step breaks probability simplex sum=1")
        self.assertTrue(torch.all(probs_next >= 0), "Euler step resulted in negative probabilities")

if __name__ == '__main__':
    unittest.main()
