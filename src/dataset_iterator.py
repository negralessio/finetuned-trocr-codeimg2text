import torch
from torch.utils.data import Dataset
from PIL import Image


class DatasetIterator(Dataset):
    """
    Class represents a dataset iterator leveraging the PyTorch data loading utility.
    See: https://pytorch.org/docs/stable/data.html
    """

    def __init__(self, df, processor, max_target_length=175):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['line_img_path'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")#.resize((384, 384))  # Do we have to resize too? A.
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
