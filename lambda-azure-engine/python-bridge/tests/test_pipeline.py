import unittest

from lambda_engine import EngineConfig, LambdaAzureEngine


class TestPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        config = EngineConfig(d_model=32, n_experts=2, n_layers=1, use_cuda=False, seed=1)
        engine = LambdaAzureEngine(config)
        output = engine.generate("test prompt", max_len=4)
        self.assertIn("Simulated output", output)


if __name__ == "__main__":
    unittest.main()
