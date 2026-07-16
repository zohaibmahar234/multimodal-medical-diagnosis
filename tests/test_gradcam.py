import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for Grad-CAM tests")
class TestGradCAM(unittest.TestCase):
    def test_generates_normalized_heatmap(self):
        from multimodal_medical_diagnosis import GradCAM, MultiModalPneumoniaModel

        model = MultiModalPneumoniaModel(clinical_input_dim=5, num_classes=2)
        model.eval()
        gradcam = GradCAM(model, model.image_encoder.features[6])

        image = torch.randn(2, 1, 128, 128, requires_grad=True)
        clinical = torch.randn(2, 5)

        heatmap = gradcam.generate(image=image, clinical=clinical)

        self.assertEqual(tuple(heatmap.shape), (2, 32, 32))
        self.assertTrue(torch.isfinite(heatmap).all().item())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        gradcam.close()


if __name__ == "__main__":
    unittest.main()
