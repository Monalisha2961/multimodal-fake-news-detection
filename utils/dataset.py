# utils/dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset for Fakeddit — handles text, image, and comments
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from torchvision import transforms

from config import (
    ROBERTA_MODEL, MAX_TEXT_LEN, MAX_COMMENT_LEN,
    IMAGE_SIZE, IMAGE_DIR, LABEL_COL
)


# ─── Image transforms ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),  # ImageNet stats
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Blank image for missing images
BLANK_IMAGE = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)


class FakedditDataset(Dataset):
    """
    Loads Fakeddit TSV rows and returns:
      - text tokens  (headline + body)
      - comment tokens
      - image tensor
      - label
    """

    def __init__(self, tsv_path: str, image_dir: str, split: str = "train", max_samples: int = None):
        self.df        = pd.read_csv(tsv_path, sep="\t")
        self.image_dir = image_dir
        self.split     = split
        self.transform = train_transforms if split == "train" else val_transforms
        self.tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

        # Drop rows with missing labels
        self.df = self.df.dropna(subset=[LABEL_COL]).reset_index(drop=True)

        # ── Limit dataset size (great for development on laptops) ─────────────
        if max_samples is not None:
            # Keep balanced classes — equal real/fake samples
            per_class = max_samples // 2
            df_balanced = []
            for label in self.df[LABEL_COL].unique():
                df_balanced.append(
                    self.df[self.df[LABEL_COL] == label].head(per_class)
                )
            self.df = pd.concat(df_balanced).sample(frac=1).reset_index(drop=True)
            print(f"[{split}] Using {len(self.df)} samples (limited & balanced)")
        
        print(f"[{split}] Loaded {len(self.df)} samples | Label distribution:\n"
              f"{self.df[LABEL_COL].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    def _tokenize(self, text: str, max_len: int):
        """Returns input_ids and attention_mask as tensors."""
        text = str(text) if not pd.isna(text) else ""
        enc  = self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def _load_image(self, post_id: str):
        """Load image or return blank tensor if missing."""
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(self.image_dir, f"{post_id}{ext}")
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    return self.transform(img)
                except Exception:
                    break
        return BLANK_IMAGE  # ~20% of Fakeddit images are missing

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Text: combine title + clean_title
        title   = str(row.get("title", ""))
        clean   = str(row.get("clean_title", ""))
        text    = title + " " + clean
        text_ids, text_mask = self._tokenize(text, MAX_TEXT_LEN)

        # ── Comments
        comment = str(row.get("top_comments", ""))
        com_ids, com_mask = self._tokenize(comment, MAX_COMMENT_LEN)

        # ── Image
        post_id = str(row.get("id", ""))
        image   = self._load_image(post_id)

        # ── Label
        label = int(row[LABEL_COL])

        return {
            "text_ids"   : text_ids,
            "text_mask"  : text_mask,
            "com_ids"    : com_ids,
            "com_mask"   : com_mask,
            "image"      : image,
            "label"      : torch.tensor(label, dtype=torch.long),
        }