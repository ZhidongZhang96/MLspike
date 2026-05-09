import unittest

import numpy as np

from mlspike import (
    spk_autocalibration,
    spk_autosigma,
    spk_calcium,
    spk_gentrain,
    tps_mlspikes,
)


class TestMLspikePython(unittest.TestCase):
    def test_spk_gentrain_fix_rate(self):
        spikes = spk_gentrain(2.0, 5.0, mode="fix-rate")
        self.assertTrue(np.all(spikes >= 0))
        self.assertTrue(np.all(spikes <= 5.0))

    def test_spk_calcium_baseline(self):
        par = spk_calcium("par", 0.1, 1.0)
        F, F0, drift = spk_calcium([], par)
        self.assertEqual(F.shape[0], 10)
        self.assertTrue(np.allclose(F, F0))
        self.assertTrue(np.allclose(drift, 0))

    def test_tps_mlspikes_shapes(self):
        spikes = np.zeros(100)
        spikes[::10] = 1
        par = tps_mlspikes("par")
        par["dt"] = 0.1
        par["finetune"]["sigma"] = 0.01
        F = tps_mlspikes(spikes, par)
        n, fit, drift, _ = tps_mlspikes(F, par)
        self.assertEqual(len(n), len(F))
        self.assertEqual(len(fit), len(F))
        self.assertEqual(len(drift), len(F))

    def test_spk_autosigma_positive(self):
        noise = np.random.randn(1000)
        sigma = spk_autosigma(noise, 0.1, "white")
        self.assertGreaterEqual(sigma, 0)

    def test_spk_autocalibration_sigmaonly(self):
        noise = np.random.randn(2000) + 2.0
        sigma = spk_autocalibration(noise, 0.1, sigmaonly=True)
        self.assertGreaterEqual(sigma, 0)


if __name__ == "__main__":
    unittest.main()
