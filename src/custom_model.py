import pandas as pd
import numpy as np
import json
import os
import requests
from pathlib import Path
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm.notebook import tqdm
from data_extraction import DataExtractor
from sklearn.model_selection import train_test_split
from dataset_iterator import DatasetIterator
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate
from datasets import load_metric
import evaluate
import torch


class CustomTrOCR:
    """
    Our custom model based on TrOCR. Contains methods for training, evaluating, and predicting.
    """

    def __init__(self):
        """
        Initialize CustomTrOCR.
        """
        self.data: pd.DataFrame = None
        self.test_set: pd.DataFrame = None      # Test Set (Hold Out Set) after fitting
        self.processor: TrOCRProcessor = None
        self.ved_model: VisionEncoderDecoderModel = None
        self.chinese_letter_count: int = 0
        self.history_loss: list = []
        self.history_cer: list = []

    def get_data(self) -> pd.DataFrame:
        """
        Return the dataframe containing the data.
        """
        return self.data

    def get_processor(self) -> TrOCRProcessor:
        """
        Return the TrOCRProcessor.
        """
        return self.processor

    def get_ved_model(self) -> VisionEncoderDecoderModel:
        """
        Return the VisionEncoderDecoderModel.
        """
        return self.ved_model

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load the data using the from_json method of the DataExtractor class.
        :param data_path: The path to the data (json-file).
        :return: The data as dataframe.
        """

        # Instantiate DataExtractor
        de = DataExtractor()

        # Check if data_path ends with .json, if not, return an error
        if not data_path.endswith(".json"):
            raise ValueError("The data_path must end with .json")

        # Check if path exists, if not, return an error
        if not os.path.exists(data_path):
            raise ValueError(f"The data_path {data_path} does not exist")

        # Get directory path & file name
        dir_path = Path(data_path).parent
        file_name = Path(data_path).name

        # Load the data
        self.data = de.from_json(attr="lines_data", file_name=file_name, from_dir=dir_path)

        return self.data

    def prepare_data(self, outlier_threshold: int = 200, strip_whitespaces: bool = True) -> pd.DataFrame:
        """
        Prepares the data for training the OCR Engine.
        Namely, keeps the relevant columns (see KEEP_LIST) and removes outlier with Threshold Param
        :param outlier_threshold: Removes rows where text_size is greater than the threshold
        :param strip_whitespaces: Bool whether or not white spaces in target column should be stripped
        :return: Cleaned DataFrame
        """

        assert len(self.data) != 0, "Load Data first!"

        # Filter only on relevant columns for training
        KEEP_LIST = ["line_img_path", "font", "theme", "language", "line_number", "text"]
        data = self.data.copy()
        data = data[KEEP_LIST]

        # Save size before removals
        n_prior = len(data)

        # Add column of text size to df (including stripping of whitespaces)
        if strip_whitespaces:
            data["text_size"] = data["text"].apply(lambda x: len(str(x).strip()))
        else:
            data["text_size"] = data["text"].apply(lambda x: len(str(x)))

        # Remove Nones in line_img_path
        data = data.dropna(axis=0)
        print(f"Number of removed None values in line_img_path: {n_prior - len(data)}")

        # Remove rows where text_size == 0 (no empty images)
        n_prior_bigger_0 = len(data)
        data = data[data.text_size > 0]
        print(f"Removed rows where text_size <= 0. Total: {n_prior_bigger_0 - len(data)}")

        # Remove outlier and print information about removal
        n_prio_or = len(data)
        data = data[data.text_size <= outlier_threshold]
        print(f"Shape of cleaned data set {data.shape}. Removed rows: {n_prio_or - len(data)}. " +
              f"Outlier Threshold: {outlier_threshold}. Strip Whitespaces: {str(strip_whitespaces)}")

        # Function to flag chinese letter
        def contains_chinese(string):
            """
            Function takes a string as input and returns True if any of the characters in the input string
             have an ASCII code within the range of Chinese ASCII symbols (0x4E00 to 0x9FFF), and False otherwise
            :param string: String to check
            :return: Bool whether contains chinese letter (True) or not (False)
            """
            for character in string:
                if 0x4E00 <= ord(character) <= 0x9FFF:
                    self.chinese_letter_count += 1
                    return True
            return False

        # Call above function to set a flag whether text contains chinese letters
        data["contains_chinese"] = data["text"].apply(lambda x: contains_chinese(x))
        print(f"Flagged {self.chinese_letter_count} rows as text that contains at least one chinese letter.")

        # Remove rows where line_number < 0
        n_prior_line_number = len(data)
        data = data[data.line_number >= 0]
        print(f"Removed rows where line_number < 0: {n_prior_line_number - len(data)}")

        # Return cleaned dataframe
        self.data = data
        return self.data

    def load_processor(self, model_path: str, source: str = "local", return_model: bool = False):
        """
        Load specifically the processor for the TrOCR-model.
        :param model_path: The path to the model (if source=="local) or model name (if source=="huggingface").
        :param source: The source of the model. Either "local" or "huggingface". Default is "local".
        :param return_model: Whether to return the model or not. Default is False.
        :return: The processor for the model.
        """

        # If source is local, check if path exists, if not, return an error
        if source == "local":
            if not os.path.exists(model_path):
                raise ValueError(f"The model_path {model_path} does not exist")

        # If source is huggingface, link to model exists on huggingface, if not, return an error
        elif source == "huggingface":
            if requests.head(f"https://huggingface.co/{model_path}").status_code != 200:
                raise ValueError(f"The model {model_path} does not exist on huggingface")


        # If source is neither local nor huggingface, return an error
        else:
            raise ValueError(f"The source {source} is not supported")

        # Load the processor
        self.processor = TrOCRProcessor.from_pretrained(model_path)

        # If return_model is True, return the processor
        if return_model:
            return self.processor

    def load_visual_encoder_decoder(self, model_path: str, source: str = "local", return_model: bool = False):
        """
        Load specifially the visual encoder decoder model.
        :param model_path: The path to the model (if source=="local) or model name (if source=="huggingface").
        :param source: The source of the model. Either "local" or "huggingface". Default is "local".
        :param return_model: Whether to return the model or not. Default is False.
        :return: The model.
        """

        # If source is local, check if path exists, if not, return an error
        if source == "local":
            if not os.path.exists(model_path):
                raise ValueError(f"The model_path {model_path} does not exist")

        # If source is huggingface, link to model exists on huggingface, if not, return an error
        elif source == "huggingface":
            if requests.head(f"https://huggingface.co/{model_path}").status_code != 200:
                raise ValueError(f"The model {model_path} does not exist on huggingface")

        # If source is neither local nor huggingface, return an error
        else:
            raise ValueError(f"The source {source} is not supported")

        # Load the ved_model
        self.ved_model = VisionEncoderDecoderModel.from_pretrained(model_path)

        # If return_model is True, return the ved_model
        if return_model:
            return self.ved_model

    def compute_cer(self, pred_ids, label_ids):
        """
        Method to compute the Character Error Rate.
        See: https://huggingface.co/spaces/evaluate-metric/cer
        :param pred_ids: IDs of our prediction
        :param label_ids: IDs of our labels
        :return: Character Error Rate (CER)
        """
        cer_metric = evaluate.load("cer")

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return cer

    def custom_train_test_split(self, df: pd.DataFrame, val_size: float, test_size: float):
        """
        Function to create custom train, val, test split created for isolating certain themes / fonts.
        :param df: Pandas Dataframe
        :param val_size: Validation size
        :param test_size: Test size
        :return:
        """

        # Sort dataframe ascending by theme to make sure that we isolate certain themes
        df = df.copy()
        df.sort_values(by="theme", ascending=True, inplace=True)

        # Take first 10% of the data for our hold out set while isolating certain themes
        k = int(len(df) * test_size)
        test_set = df[:k]

        # Remove test set from df for splitting
        df = df[k+1:]
        train_set, val_set = train_test_split(df, test_size=val_size)

        return train_set, val_set, test_set


    def train_model(self, test_size: float = 0.1, val_size: float = 0.1, batch_size: int = 16, shuffle: bool = True, epochs: int = 100,
                    save_dir: str = "../models/trocr_all_v1",
                    take_subsample: bool = False, subsample_size: float = 0.1,
                    remove_chinese_letter: bool = True,
                    use_custom_train_test_split: bool = True):
        """
        Method to finetune the corresponding model
        :param test_size: Size of the test df
        :param batch_size: Batch size
        :param shuffle: Bool whether to shuffle
        :param epochs: Number of epochs of the training
        :param save_dir: Directory to save finetuned model
        :param take_subsample: Whether to train on a subsample
        :param subsample_size: Size of the subsample
        :param remove_chinese_letter: Bool whether to remove rows that contain at least one chinese letter
        :return:
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ved_model.to(device)
        print(f"device: {device}")

        data = self.data.copy()

        # Take subsample of data if it is set to True
        if take_subsample:
            data = data.sample(frac=subsample_size)

        # Remove rows that contain chinese letter
        if remove_chinese_letter:
            data = data[data.contains_chinese == False]

        # Split data into train and test (validation) dataframe
        if use_custom_train_test_split:
            train_df, val_df, self.test_set = self.custom_train_test_split(df = data,
                                                                           val_size=val_size,
                                                                           test_size=test_size)
        else:
            # Hold out set
            temp, self.test_set = train_test_split(data, test_size=test_size)
            # Train Validation Set
            train_df, val_df = train_test_split(temp, test_size=val_size)

        # Save train, val and test set for eda later
        if not os.path.exists("../data/extracted/training/" + save_dir.split("/")[-1]):
            os.makedirs("../data/extracted/training/" + save_dir.split("/")[-1])

        train_df.to_json("../data/extracted/training/" + save_dir.split("/")[-1] + "/train_df")
        val_df.to_json("../data/extracted/training/" + save_dir.split("/")[-1] + "/val_df")
        self.test_set.to_json("../data/extracted/training/" + save_dir.split("/")[-1] + "/test_df")

        # Keep only for training relevant columns
        train_df = train_df[["line_img_path", "text"]]
        val_df = val_df[["line_img_path", "text"]]

        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        # Information about the datasets
        print("Shape of training examples:", train_df.shape)
        print("Shape of validation examples:", val_df.shape)

        # Instantiate Dataset Iterator with corresponding dataframes and processor
        train_dataset = DatasetIterator(df=train_df, processor=self.processor)
        eval_dataset = DatasetIterator(df=val_df, processor=self.processor)

        # Verify Example:
        encoding = train_dataset[0]
        for k, v in encoding.items():
            print(k, v.shape)

        # Use Dataloader from pytorch
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

        # Set special tokens used for creating the decoder_input_ids from the labels
        # See: https://huggingface.co/transformers/v4.12.5/model_doc/trocr.html
        self.ved_model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.ved_model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.ved_model.config.vocab_size = self.ved_model.config.decoder.vocab_size

        # Set beam search parameters
        self.ved_model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.ved_model.config.max_new_tokens = 64
        self.ved_model.config.early_stopping = True
        self.ved_model.config.no_repeat_ngram_size = 3
        self.ved_model.config.length_penalty = 2.0
        self.ved_model.config.num_beams = 4

        optimizer = torch.optim.AdamW(self.ved_model.parameters(), lr=5e-5)

        for epoch in range(epochs):  # loop over the dataset multiple times
            # train
            self.ved_model.train()
            train_loss = 0.0
            for batch in tqdm(train_dataloader):
                # get the inputs
                for k, v in batch.items():
                    batch[k] = v.to(device)

                # forward + backward + optimize
                outputs = self.ved_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))

            # evaluate
            self.ved_model.eval()
            valid_cer = 0.0
            with torch.no_grad():
                for batch in tqdm(eval_dataloader):
                    # run batch generation
                    outputs = self.ved_model.generate(batch["pixel_values"].to(device))
                    # compute metrics
                    cer = self.compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                    valid_cer += cer

            print(f"Validation CER: {valid_cer / len(eval_dataloader)}")

            # Append epoch information about loss and cer to history
            self.history_loss.append(train_loss / len(train_dataloader))
            self.history_cer.append(valid_cer / len(eval_dataloader))

            # Save model for each epoch
            self.dump_model(dir_path=save_dir + f"_ep{epoch}")

        # Save history of model
        self.save_history(model_name=save_dir)


    def save_history(self, model_name: str):
        """
        Function to save history of the model
        :param model_name: Save dir of model
        :return:
        """
        data = {"history_loss": self.history_loss,
                "history_cer": self.history_cer}
        df = pd.DataFrame(data)

        if not os.path.exists("../data/eval"):
            os.makedirs("../data/eval")

        df.to_json("../data/eval/history_" + model_name.split("/")[-1])


    def load_model(self, model_path: str, source: str = "local") -> None:
        """
        Load the model (including both TrOCRProcessor & VisualEncoderDecoderModel).
        :param model_path: The path to the model (if source=="local) or model name (if source=="huggingface").
        :param source: The source of the model. Either "local" or "huggingface". Default is "local".
        :return: None.
        """

        # Load the processor
        self.load_processor(model_path=model_path, source=source)

        # Load the model
        self.load_visual_encoder_decoder(model_path=model_path, source=source)

    def dump_model(self, dir_path: str):
        """
        Dump the model and processor to the path.
        :param dir_path: The dir_path to the directory where the model and processor will be saved. If the directory does not exist, it will be created.
        """

        # Check if dir_path exists, if not, create it
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Save the model & processor
        self.ved_model.save_pretrained(dir_path)
        self.processor.save_pretrained(dir_path)

    def list_current_models(self):
        """
        Print the name of all current models in the models directory
        :return:
        """
        # Directory where models are saved
        models_dir = "../models"

        # Print all models
        for name in os.listdir(models_dir):
            print(models_dir + "/" + name)

    def predict_single(self, image_path: str, return_dict: bool = False):
        """
        Predict the text of a single image.
        :param image_path: The path to the image.
        :param return_dict: Whether to return a dictionary {generated_text, generated_ids, pixel_values, img_path} or just a str containing generated_text. Default is False.
        :return: The predicted text as string or dictionary.
        """

        # Check if image_path exists, if not, return an error
        if not os.path.exists(image_path):
            raise ValueError(f"The image_path {image_path} does not exist")

        # Check if image_path ends with .jpg, .jpeg, .png, .tif, .tiff, if not, return an error
        if not image_path.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            raise ValueError(f"The image_path {image_path} must end with .jpg, .jpeg, .png, .tif, or .tiff")

        # Load the image
        image = Image.open(image_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ved_model.to(device)

        # Get pixel values
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate ids & text
        generated_ids = self.ved_model.generate(pixel_values).to(device)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # If return_dict is True, return a dictionary
        if return_dict:
            return {"generated_text": generated_text, "generated_ids": generated_ids, "pixel_values": pixel_values,
                    "image_path": image_path}

        # If return_dict is False, return the text
        else:
            return generated_text

    def predict_batch(self, image_paths: list[str], return_dict: bool = False) -> list:
        """
        Predict the text of a batch of images.
        :param image_paths: The paths to the images.
        :param return_dict: Whether to return a dictionary {generated_text, generated_ids, pixel_values, img_path} or just a str containing generated_text for each prediction. Default is False.
        :return: The predicted text as list of strings or dictionaries.
        """

        # Create a list to store the predictions
        pred_list = []

        # Iterate over all image_paths and predict the text
        for image_path in image_paths:
            pred = self.predict_single(image_path=image_path, return_dict=return_dict)
            pred_list.append(pred)

        # Return the list of predictions (either as strings or dictionaries)
        return pred_list

    def predict(self) -> pd.DataFrame:
        """
        Predict the text of all entries of the dataframe within the data attribute.
        :return: The dataframe within the data attribute with the predicted text added as a new column.
        """

        # Check if data attribute is None, if so, return an error
        if self.data is None:
            raise ValueError("The data attribute is None. Please use load_data() first.")

        # Check if dataframe contains a column "line_img_path", if not, return an error
        if "line_img_path" not in self.data.columns:
            raise ValueError("The dataframe does not contain a column 'line_img_path'.")

        # Iterate over all entries of the dataframe, predict the text and add it to the dataframe
        for index, row in self.data.iterrows():

            # If the image_path is not None, predict the text, if not, set the text to None
            if row["line_img_path"] is not None:
                generated_text = self.predict_single(row["line_img_path"], return_dict=False)
            else:
                generated_text = None

            # Add the predicted text to the dataframe
            self.data.loc[index, "generated_text"] = generated_text

        # Return the dataframe
        return self.data
