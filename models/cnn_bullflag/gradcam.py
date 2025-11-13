from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


class GradCAM1D:
    """
    Grad-CAM for 1D CNNs targeting a named convolutional layer.

    Assumptions:
    - model(x) returns a dict with keys like "flag", "breakout", "retest", "continuation",
      each being a scalar tensor or shape [B, 1].
    - For target_head == "sequence", we backprop from the geometric mean across the four heads.
    - Target layer (e.g., "conv4") is the last conv before any global pooling/flattening.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        modules = dict(self.model.named_modules())
        if target_layer_name not in modules:
            raise KeyError(f"Target layer '{target_layer_name}' not found in model.named_modules().")
        self.target_layer = modules[target_layer_name]

        self.activations: Optional[torch.Tensor] = None  # [B, C, L_conv]
        self.gradients: Optional[torch.Tensor] = None    # [B, C, L_conv]

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> None:
            # Save the forward activations from the target layer
            self.activations = output

        def backward_hook(module: torch.nn.Module, grad_input: Any, grad_output: Any) -> None:
            # grad_output is a tuple; for conv layers, first element corresponds to dL/dA
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        # Use full backward hook (non-deprecated) to catch gradients reliably
        self.target_layer.register_full_backward_hook(backward_hook)

    @torch.no_grad()
    def _maybe_eval(self) -> None:
        # Grad-CAM should typically run in eval mode to freeze BN/dropout statistics.
        self.model.eval()

    def generate_cam(self, x: torch.Tensor, target_head: str = "sequence") -> torch.Tensor:
        """
        x: [1, n_features, seq_len] on any device (moved internally)
        target_head: which head's output to explain; 'sequence' uses geomean across four heads.
        returns: cam_upsampled: [seq_len] tensor with values in [0, 1] on CPU
        """
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected x shape [1, C, L]; got {tuple(x.shape)}")

        self._maybe_eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        x = x.to(self.device)

        # Forward pass WITH gradients (do not use torch.no_grad())
        outputs: Dict[str, torch.Tensor] = self.model(x)  # type: ignore[assignment]

        def _as_scalar(t: torch.Tensor) -> torch.Tensor:
            if t.ndim == 0:
                return t
            return t.view(-1)[0]

        if target_head == "sequence":
            # Geometric mean of four head probabilities (or scores)
            try:
                flag = _as_scalar(outputs["flag"])  # type: ignore[index]
                breakout = _as_scalar(outputs["breakout"])  # type: ignore[index]
                retest = _as_scalar(outputs["retest"])  # type: ignore[index]
                continuation = _as_scalar(outputs["continuation"])  # type: ignore[index]
            except KeyError as e:
                raise KeyError(f"Missing head in model outputs: {e}")
            # Multiply then 4th root; add tiny epsilon for numerical stability
            eps = 1e-8
            target = (flag + eps) * (breakout + eps) * (retest + eps) * (continuation + eps)
            target = target.pow(0.25)
        else:
            if target_head not in outputs:
                raise KeyError(f"Unknown target_head '{target_head}' in model outputs {list(outputs.keys())}")
            target = _as_scalar(outputs[target_head])

        # Backprop from scalar target
        self.model.zero_grad(set_to_none=True)
        target.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # activations: [1, C, L_conv], gradients: [1, C, L_conv]
        activations = self.activations[0]  # [C, L_conv]
        gradients = self.gradients[0]      # [C, L_conv]

        # Global-average pool gradients over time -> weights [C]
        weights = gradients.mean(dim=1)

        # Weighted sum over channels -> cam [L_conv]
        cam = torch.zeros(activations.shape[1], device=self.device)
        cam = cam + torch.einsum("c,cl->l", weights, activations)

        # ReLU and normalize to [0,1]
        cam = F.relu(cam)
        max_val = cam.max()
        if torch.isfinite(max_val) and max_val > 0:
            cam = cam / max_val

        # Upsample to original seq_len
        cam = cam.view(1, 1, -1)
        cam_upsampled = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[0, 0]

        # Ensure CPU tensor
        return cam_upsampled.detach().cpu()

