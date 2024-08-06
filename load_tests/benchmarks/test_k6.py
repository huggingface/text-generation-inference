import os
from unittest import TestCase

from benchmarks.k6 import K6RampingArrivalRateExecutor, K6Config, K6ConstantVUsExecutor, K6Benchmark, ExecutorInputType


class K6RampingArrivalRateExecutorTest(TestCase):
    def test_render(self):
        executor = K6RampingArrivalRateExecutor(
            100,
            1,
            "1s",
            [
                {"target": 1, "duration": "30s"},
                {"target": 100, "duration": "30s"}
            ],
            ExecutorInputType.SHAREGPT_CONVERSATIONS)
        executor.render()
        self.assertIsNotNone(executor.rendered_file)
        with open(executor.rendered_file, "r") as f:
            content = f.read()
            self.assertTrue("stages: [" in content)
            self.assertTrue("target: 1, duration: '30s'" in content)
            self.assertTrue(os.getcwd() in content)


class K6BenchmarkTest(TestCase):
    def test_prepare_inputs(self):
        executor = K6ConstantVUsExecutor(1, '1m')
        config = K6Config("test", executor, input_num_tokens=500)
        bench = K6Benchmark(config, "output")
