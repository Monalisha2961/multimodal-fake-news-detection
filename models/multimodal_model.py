# models/multimodal_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Architecture: RoBERTa (text) + ResNet-50 (image) + RoBERTa (comments)
#               → Concatenation Fusion → Classifier
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision import models

from config import (
    ROBERTA_MODEL, TEXT_FEAT_DIM, IMAGE_FEAT_DIM,
    COMMENT_FEAT_DIM, FUSED_DIM, NUM_CLASSES, DROPOUT
)


class TextEncoder(nn.Module):
    """RoBERTa encoder — shared for both text and comments."""

    def __init__(self, freeze_layers: int = 8):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(ROBERTA_MODEL)

        # Freeze the first N transformer layers to save memory & speed up training
        for i, layer in enumerate(self.roberta.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        return out.last_hidden_state[:, 0, :]   # (B, 768)


class ImageEncoder(nn.Module):
    """ResNet-50 encoder — removes final FC layer, returns 2048-dim features."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)

        # Freeze early layers (layer1 & layer2)
        for name, param in resnet.named_parameters():
            if "layer1" in name or "layer2" in name or "conv1" in name or "bn1" in name:
                param.requires_grad = False

    def forward(self, images):
        feat = self.backbone(images)       # (B, 2048, 1, 1)
        return feat.view(feat.size(0), -1) # (B, 2048)


class MultimodalFakeNewsDetector(nn.Module):
    """
    Full multimodal model:
      TextEncoder(title)  → 768
      ImageEncoder        → 2048
      TextEncoder(comment)→ 768
      ─────────────────────────
      Concatenate         → 3584
      FC layers + Dropout → NUM_CLASSES
    """

    def __init__(self):
        super().__init__()

        # Encoders
        self.text_encoder    = TextEncoder(freeze_layers=8)
        self.image_encoder   = ImageEncoder()
        self.comment_encoder = TextEncoder(freeze_layers=8)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(FUSED_DIM, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, text_ids, text_mask, com_ids, com_mask, images):
        text_feat    = self.text_encoder(text_ids, text_mask)       # (B, 768)
        image_feat   = self.image_encoder(images)                   # (B, 2048)
        comment_feat = self.comment_encoder(com_ids, com_mask)      # (B, 768)

        # Concatenation fusion
        fused = torch.cat([text_feat, image_feat, comment_feat], dim=1)  # (B, 3584)

        logits = self.classifier(fused)   # (B, NUM_CLASSES)
        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = MultimodalFakeNewsDetector()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")
