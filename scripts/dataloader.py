# dataloader.py

from torch.utils.data import Dataset, DataLoader
import torch

class TextDescriptionDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): A dictionary where keys are text descriptions and values are lists,
                         with the first element being the positive description and the rest being negative samples.
        """
        self.samples = []
        for text, descriptions in data.items():
            pos_description = descriptions[0]
            neg_descriptions = descriptions[1:]
            for neg_description in neg_descriptions:
                if text and pos_description and neg_description:
                    text = text.split(';;;')[-1]
                    pos_description = pos_description.split(';;;')[-1]
                    neg_description = neg_description.split(';;;')[-1]
                    self.samples.append((text, pos_description, neg_description))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_description, description_pos, description_neg = self.samples[idx]
        return {
            "text_description": text_description,
            "description_pos": description_pos,
            "description_neg": description_neg
        }

def create_dataloader(data, batch_size=8, shuffle=True):
    dataset = TextDescriptionDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
