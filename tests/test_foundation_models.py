from __future__ import annotations

import unittest

from agents.vision.foundation_models import get_model_spec, list_model_specs


class TestFoundationModelRegistry(unittest.TestCase):
    def test_registry_includes_new_encoder_specs(self) -> None:
        specs = {spec.name: spec for spec in list_model_specs()}
        self.assertIn("gigapath", specs)
        self.assertIn("hoptimus0", specs)

    def test_gigapath_spec_matches_expected_metadata(self) -> None:
        spec = get_model_spec("gigapath")
        self.assertEqual(spec.embed_dim, 1536)
        self.assertTrue(spec.gated)
        self.assertEqual(spec.hub, "prov-gigapath/prov-gigapath")

    def test_hoptimus0_spec_matches_expected_metadata(self) -> None:
        spec = get_model_spec("hoptimus0")
        self.assertEqual(spec.embed_dim, 1536)
        self.assertTrue(spec.gated)
        self.assertEqual(spec.hub, "bioptimus/H-optimus-0")

    def test_virchow2_timm_kwargs_include_required_init(self) -> None:
        spec = get_model_spec("virchow2")
        self.assertIsNotNone(spec.timm_kwargs)
        assert spec.timm_kwargs is not None
        self.assertEqual(spec.timm_kwargs.get("mlp_layer"), "SwiGLUPacked")
        self.assertEqual(spec.timm_kwargs.get("act_layer"), "SiLU")
        self.assertEqual(spec.timm_kwargs.get("init_values"), 1e-5)


if __name__ == "__main__":
    unittest.main()
