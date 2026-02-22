from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """
    Dataset format (CSV):
    image_a,image_b,label
    path/to/original.png,path/to/generated.png,1
    """

    def __init__(self, csv_path: str, transform=None):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        required = {"image_a", "image_b", "label"}
        if not required.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must contain columns: {required}")

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_a = Image.open(row["image_a"]).convert("RGB")
        image_b = Image.open(row["image_b"]).convert("RGB")
        label = float(row["label"])

        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        return image_a, image_b, label
