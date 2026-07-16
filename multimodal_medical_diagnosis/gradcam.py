"""Grad-CAM utilities for multimodal chest X-ray models."""

from __future__ import annotations

import torch


class GradCAM:
    """Computes Grad-CAM heatmaps for the image branch of the multimodal model."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image: torch.Tensor,
        clinical: torch.Tensor,
        class_index: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image, clinical)

        if class_index is None:
            class_index = logits.argmax(dim=1)
        if isinstance(class_index, int):
            class_index = torch.full((logits.size(0),), class_index, device=logits.device)

        selected = logits.gather(1, class_index.view(-1, 1)).sum()
        selected.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1)
        cam = torch.relu(cam)
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam

    def close(self) -> None:
        self._forward_hook.remove()
        self._backward_hook.remove()
