2. Grad-CAM for 1D CNN (activation map for visualisation)

Goal: for a given input window x:

Get a time-indexed importance score for each step (len = 200)

Use it to produce activation_map.indices and activation_map.intensities in your CNNOutput JSON

2.1 Concept

For a given head (say sequence_score or flag_head):

Take activations A from the last conv layer: shape [C, L_conv]

Take gradients dY/dA for the head output w.r.t A: [C, L_conv]

Compute channel weights: w_c = mean_t(grad[c, t])

Compute CAM: cam[t] = ReLU( sum_c w_c * A[c, t] )

Normalise cam to [0,1]

Upsample from length L_conv to original seq_len = 200 (because of pooling)

That gives you a 1D importance profile.

2.2 GradCAM1D utility class spec

In models/cnn_bullflag/gradcam.py:

import torch
import torch.nn.functional as F
from typing import Dict

class GradCAM1D:
    def __init__(self, model, target_layer_name: str, device: str = "cpu"):
        """
        model: BullFlagCNNMultiHead
        target_layer_name: e.g. "conv4" (the last conv block/layer)
        """
        self.model = model
        self.device = device
        self.target_layer = dict(model.named_modules())[target_layer_name]

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output: [B, C, L_conv]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0]: [B, C, L_conv]
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(
        self,
        x: torch.Tensor,
        target_head: str = "sequence"
    ) -> torch.Tensor:
        """
        x: [1, n_features, seq_len]
        target_head: which head's output to explain.
                     For v1, you can: sequence = geometric mean of all heads.
        returns: cam_upsampled: [seq_len] tensor with values in [0, 1]
        """
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        x = x.to(self.device)
        outputs = self.model(x)  # dict of heads

        if target_head == "sequence":
            # geometric mean of probs
            probs = torch.cat([
                outputs["flag"],
                outputs["breakout"],
                outputs["retest"],
                outputs["continuation"]
            ], dim=1)  # [1, 4]
            target = (probs.prod(dim=1) ** 0.25)  # [1]
        else:
            target = outputs[target_head].view(-1)  # [1]

        # Backprop from target scalar
        target.backward()

        # activations: [1, C, L_conv]
        # gradients:   [1, C, L_conv]
        activations = self.activations[0]   # [C, L_conv]
        gradients = self.gradients[0]       # [C, L_conv]

        # Global-average pool gradients over time: w_c
        weights = gradients.mean(dim=1)     # [C]

        # Weighted sum over channels
        cam = torch.zeros(activations.shape[1], device=self.device)  # [L_conv]
        for c in range(activations.shape[0]):
            cam += weights[c] * activations[c]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to original seq_len
        cam = cam.view(1, 1, -1)   # [1, 1, L_conv]
        cam_upsampled = F.interpolate(
            cam,
            size=x.shape[-1],  # seq_len
            mode="linear",
            align_corners=False
        )[0, 0]  # [seq_len]

        return cam_upsampled.detach().cpu()


This class will:

Hook into the last conv

Let you call generate_cam(x, target_head="sequence")

Return a [seq_len] vector of importance scores in [0,1]

2.3 Integrate Grad-CAM into inference

In infer.py, extend BullFlagCNNInfer:

from .gradcam import GradCAM1D

class BullFlagCNNInfer:
    def __init__(..., enable_gradcam: bool = False):
        ...
        self.enable_gradcam = enable_gradcam
        if enable_gradcam:
            self.gradcam = GradCAM1D(self.model, target_layer_name="conv4", device=device)

    def __call__(self, feature_window: np.ndarray) -> Dict[str, Any]:
        ...
        with torch.no_grad():
            outputs = self.model(x.to(self.device))
            flag = outputs["flag"].item()
            breakout = outputs["breakout"].item()
            retest = outputs["retest"].item()
            continuation = outputs["continuation"].item()

        sequence_score = (flag * breakout * retest * continuation) ** 0.25

        activation_map = None
        if self.enable_gradcam:
            # Need a forward+backward pass WITH gradients
            x_grad = x.clone().to(self.device).requires_grad_(True)
            cam = self.gradcam.generate_cam(x_grad, target_head="sequence")  # [seq_len]
            activation_map = cam.numpy().tolist()  # length 200

        return {
            "flag_prob": flag,
            "breakout_prob": breakout,
            "retest_prob": retest,
            "continuation_prob": continuation,
            "sequence_score": sequence_score,
            "activation_map": activation_map,  # or None
        }


Then you map:

"activation_map": {
  "indices": [0, 1, 2, ..., 199],
  "intensities": [cam[0], cam[1], ..., cam[199]]
}


Directly into your CNNOutput JSON.