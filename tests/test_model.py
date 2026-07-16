import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for multimodal model tests")
class TestMultiModalPneumoniaModel(unittest.TestCase):
    def test_forward_fuses_image_and_clinical_data(self):
        from multimodal_medical_diagnosis import MultiModalPneumoniaModel

        model = MultiModalPneumoniaModel(clinical_input_dim=8, num_classes=2)
        image = torch.randn(4, 1, 128, 128)
        clinical = torch.randn(4, 8)

        logits = model(image, clinical)

        self.assertEqual(tuple(logits.shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
