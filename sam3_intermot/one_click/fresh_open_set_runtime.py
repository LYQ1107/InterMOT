"""Strict R2 current-identity predictor: no tracker or association authority."""
from pathlib import Path
import numpy as np
import torch
from .intervention_features import FEATURE_NAMES, feature_vector
from .open_set_verifier import OpenSetVerifierHead, predictions


class FreshOpenSetPredictor:
    def __init__(self, checkpoint):
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if saved["schema"] != "N72R21R2_CURRENT_AXIS_OPEN_SET_V1" or saved["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Fresh R2 current-axis schema required, not historical R1 weights")
        self.selection = saved["selection"]
        if self.selection["status"] == "UNCALIBRATED_EPOCH_NOT_DEPLOYABLE":
            raise ValueError("Uncalibrated epoch may not become a selected current identity verifier")
        if saved.get("association_authority", False):
            raise ValueError("Current correctness cannot acquire future global-MOT authority")
        self.model = OpenSetVerifierHead(saved["head_family"]).eval()
        self.model.load_state_dict(saved["model"], strict=True)
        self.model.requires_grad_(False)
        self.mean, self.scale = np.asarray(saved["FIT_mean"], np.float32), np.asarray(saved["FIT_std"], np.float32)
        if self.mean.shape != (32,) or self.scale.shape != (32,) or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Finite FIT-only normalizer required")

    def predict_axis(self, axis):
        if not axis or axis[-1]["candidate_uid"] is not None:
            raise ValueError("All current candidates plus explicit NONE required")
        vectors = []
        for row in axis:
            vector = feature_vector(row["features"]).tolist()
            if vector != row["feature_vector"]:
                raise ValueError("Only the exact registered runtime feature vector may be scored")
            vectors.append(vector)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy((np.asarray(vectors, np.float32) - self.mean) / self.scale))
        return predictions(logits, self.selection["temperature"])
