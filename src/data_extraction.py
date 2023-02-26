import pandas as pd
import numpy as np
import json
import os
import cv2
from pathlib import Path
from utils import find

class DataExtractor:
    """
    Class that handles data extraction from json files.
    """



    def __init__(self, top_dir_path: str = "../data/raw"):
        """
        Initialize DataExtractor.
        """
        self.top_dir_path: str = top_dir_path
        self.data: pd.DataFrame = None
        self.lines_data: pd.DataFrame = None


    
    def get_data(self) -> pd.DataFrame:
        """
        Return the data attribute.
        """
        return self.data

    

    def get_lines_data(self) -> pd.DataFrame:
        """
        Return the lines_data attribute.
        """
        return self.lines_data



    def extract_data(self, top_dir_path: str = None) -> pd.DataFrame:
        """
        Crawl the directory structure of the project, extract all necessary data from annotation.json & info.json files.
        :param top_dir_path: The path of the top directory containing the data.
        :return: A DataFrame containing the data extracted from the json files.
        """

        # Check if top_dir_path is None
        if top_dir_path is None:
            top_dir_path = self.top_dir_path

        # Get mapping dictionaries with column names and key paths to use for extraction from json files
        annotations_mapping = {"img_path":"images.0.file_name",
                               "img_width":"images.0.width",
                               "img_height":"images.0.height",
                               "bbox":"annotations.0.bbox",
                               "char_width":"annotations.0.character_width",
                               "char_height":"annotations.0.character_height",
                               "ln_start":"annotations.0.line_number_start",
                               "ln_end":"annotations.0.line_number_end",
                               "lines_data":"annotations.0.lines_data"}

        info_mapping = {"font":"config.font",
                        "theme":"config.theme.0",
                        "timestamp":"time"}

        # Create a list of metadata to be extracted from file paths
        meta_data = ["language", "repository", "file"]

        # Combine keys into a single list and set up a dictionary to store the extracted data
        key_list = list(annotations_mapping.keys()) + list(info_mapping.keys()) + meta_data
        data_dict = {key:[] for key in key_list}

        # Get paths of annotation.json & info.json files
        p = Path(top_dir_path)
        annotation_paths = [x for x in p.rglob("annotations.json")]
        info_paths = [x for x in p.rglob("info.json") if Path.exists(x.parent.joinpath("annotations.json"))]

        # Extract data from annotation.json files
        for file_path in annotation_paths:
            with open(file_path, "r") as f:
                content = json.load(f)
                for key in annotations_mapping.keys():
                    data_dict[key].append(find(content, annotations_mapping[key]))
                # Extract metadata from file path (last 3 elements of directory path = meta_data list reversed)
                for key in meta_data:
                    data_dict[key].append(file_path.parent.parts[(meta_data[::-1].index(key)+1)*-1])

        # Extract data from info.json files
        for file_path in info_paths:
            with open(file_path, "r") as f:
                content = json.load(f)
                for key in info_mapping.keys():
                    data_dict[key].append(find(content, info_mapping[key]))


        # Add id column to data_dict
        data_dict["id"] = ["img" + str(x) for x in range(1, len(data_dict["img_path"])+1)]

        # Put top_dir_path in front of the img_path
        data_dict["img_path"] = [Path(top_dir_path).joinpath(x).as_posix() for x in data_dict["img_path"]]

        # Create a DataFrame from data_dict and set its index to id
        data_df = pd.DataFrame(data_dict)
        data_df.set_index("id", inplace=True)

        # Store df in data attribute and return
        self.data = data_df
        return data_df



    def extract_lines_data(self, remove_nan: bool = False, remove_negatives: bool = True) -> pd.DataFrame:
        """
        Extract lines data from the data attribute.
        :param remove_nan: Whether to remove rows with NaN values.
        :return: A DataFrame containing the data extracted from the lines_data column.
        """

        # Check if data attribute is None, if so raise error
        if self.data is None:
            raise ValueError("Data attribute is None. Please run extract_data() first.")

        # Create a list of metadata to be extracted from data attribute
        meta_data = ["img_path", "font", "theme", "language", "repository", "file"]

        # Create a list of lines_data to be extracted from lines_data column within data attribute
        lines_data = ["line_number", "x", "y", "height", "width", "character_width", "code_width", "text"]

        # Set up a dictionary to store the extracted data & add img_id column
        data_dict = {key:[] for key in ["img_id"] + meta_data + lines_data}
        
        # Extract data
        for index, row in self.data.iterrows():
            for line in row["lines_data"]:
                
                # Extract img_id from index for each line
                data_dict["img_id"].append(index)
                
                # Extract metadata from respective columns
                for key in meta_data:
                    data_dict[key].append(row[key])
                
                # Extract lines_data from lines_data column
                for key in lines_data:
                    data_dict[key].append(line[key])
                

        # Create a DataFrame from data_dict and set its index to id
        data_df = pd.DataFrame(data_dict)

        # Remove lines with negative x-values from annotations.json
        if remove_negatives:
            n_negatives = len(data_df[data_df.x < 0])
            data_df = data_df[data_df.x >= 0]
            print(f"Removed negatives in df_lines. Total: {n_negatives}")

        # Remove rows with NaN values if remove_nan is True
        if remove_nan:
            data_df.dropna(inplace=True)

        # Add id column to data_dict
        data_df["id"] = ["line" + str(x) for x in range(1, len(data_df)+1)]
        data_df.set_index("id", inplace=True)

        # Store df in lines_data attribute and return
        self.lines_data = data_df

        return data_df



    def generate_line_images(self, save_dir: str = "line_images", use_code_width: bool = False, overwrite: bool = False) -> pd.DataFrame:
        """
        Crop images based on lines_data attribute, save in folder specified by save_dir for each image and add the image paths to the lines_data attribute. Return the lines_data attribute.
        :param save_dir: The directory to save the cropped images to.
        :param use_code_width: Whether to use the code_width column to crop the images. If set to False, the total line width will be used instead.
        :param overwrite: Whether to overwrite existing images.
        :return: lines_data attribute as DataFrame.
        """

        # Check if lines_data attribute is None, if so raise error
        if self.lines_data is None:
            raise ValueError("Lines_data attribute is None. Please run extract_lines_data() first.")

        # Create a list to store save_paths for the cropped images
        path_list = []

        # Crop images
        for index, row in self.lines_data.iterrows():
            
            # If any of the following columns are NaN, skip the row and append None to path_list
            if any([pd.isnull(x) for x in [row["img_path"], row["x"], row["y"], row["width"], row["height"]]]):
                path_list.append(None)
                continue

            # Get image path
            img_path = Path(row["img_path"])

            # Get dir_path and file_path of line_image to be saved
            dir_path = Path(img_path.parent).joinpath(save_dir)
            file_path = Path(dir_path).joinpath("line" + str(row["line_number"]) + ".png")

            # Append file_path to path_list
            path_list.append(file_path.as_posix())

            # Create dir_path if it doesn't exist
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            else:
                # Check if file_path (line-image file) already exists
                if os.path.exists(file_path):
                    # If overwrite is True, delete existing line-image
                    if overwrite:
                        os.remove(file_path)
                    # If overwrite is False, skip to next line
                    else:
                        continue
            
            # Get image
            img = cv2.imread(img_path.as_posix())

            # Get x, y, height & width of line_img to be cropped from img
            x = row["x"]
            y = row["y"]
            height = int(row["height"])
            if use_code_width:
                width = int(row["code_width"])
            else:
                width = int(row["width"])

            # Crop image
            cropped_img = img[y:y+height, x:x+width]

            # Save image
            cv2.imwrite(file_path.as_posix(), cropped_img)

        # Append flag_list as column to df
        #self.lines_data["flag"] = flag_list

        # If it doesn't exist yet, insert path list into lines_data attribute behind img_path column. Otherwise, replace the existing column.
        if "line_img_path" not in self.lines_data.columns:
            ip_pos = self.lines_data.columns.to_list().index("img_path")
            self.lines_data.insert(ip_pos + 1, "line_img_path", path_list)
        else:
            self.lines_data["line_img_path"] = path_list
        
        # Return lines_data attribute
        return self.lines_data



    def get_line_image_paths(self, save_dir: str = "line_images") -> pd.DataFrame:
        """
        Get all line-img paths in save_dir based on df-entries img_path & line_number and set the line_img_path column in lines_data attribute to the paths of the line images.
        :param save_dir: The directory to find the line images in.
        :return: lines_data attribute as DataFrame.
        """

        # Check if lines_data attribute is None, if so raise error
        if self.lines_data is None:
            raise ValueError("Lines_data attribute is None. Please run extract_lines_data() first.")

        # Create a list to store save_paths of line_images
        path_list = []

        # Get all line_img paths in save_dir
        for index, row in self.lines_data.iterrows():
                
                # If any of the following columns are NaN, skip the row and append None to path_list
                if any([pd.isnull(x) for x in [row["img_path"], row["x"], row["y"], row["width"], row["height"]]]):
                    path_list.append(None)
                    continue

                # Get image path
                img_path = Path(row["img_path"])
    
                # Get dir_path and file_path of line_image to be saved
                dir_path = Path(img_path.parent).joinpath(save_dir)
                file_path = Path(dir_path).joinpath("line" + str(row["line_number"]) + ".png")

                # Check if dir_path exists, otherwise raise error
                if not os.path.exists(dir_path):
                    raise FileNotFoundError(f"Directory {dir_path} does not exist. Please run generate_line_images() first.")
                
                # Check if file_path exists, otherwise raise error
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File {file_path} does not exist. Please run generate_line_images() first.")
    
                # Append file_path to path_list
                path_list.append(file_path.as_posix())

        # If it doesn't exist yet, insert path list into lines_data attribute behind img_path column. Otherwise, replace the existing column.
        if "line_img_path" not in self.lines_data.columns:
            ip_pos = self.lines_data.columns.to_list().index("img_path")
            self.lines_data.insert(ip_pos + 1, "line_img_path", path_list)
        else:
            self.lines_data["line_img_path"] = path_list

        # Return lines_data attribute
        return self.lines_data



    def to_json(self, attr: str = "all", file_name = None, to_dir: str = None, overwrite: bool = True) -> None:
        """
        Save data and/or lines_data attribute (DataFrame) as json file.
        :param attr: The attribute containing the data to save as json. Can be "data", "lines_data" or "all". If set to "all", both data and lines_data attributes will be saved. Default is "all".
        :param file_name: The name of the json file to save. If None, the name of the json file will be the same set to the attr parameter. If attr parameter is set to "all", a list of file names ["data_file", "lines_data_file"] can be passed. Default is None.
        :param to_dir: The directory to save the json file to. If None, the json file will be saved to the current working directory. Default is None.
        :param overwrite: Whether to overwrite existing json file(s). Default is True.
        :return: None.
        """
        
        # Check if attr is valid
        if attr not in ["data", "lines_data", "all"]:
            raise ValueError(f"attr must be either 'data', 'lines_data' or 'all', not {attr}.")

        # Check if data or lines_data attribute is None (depending on attr parameter), if so raise error
        if self.data is None and attr == "data":
            raise ValueError("Data attribute is None. Please run from_json() or get_data() first.")
        elif self.lines_data is None and attr == "lines_data":
            raise ValueError("Lines_data attribute is None. Please run from_json() or extract_lines_data() first.")
        elif self.data is None and self.lines_data is None and attr == "all":
            raise ValueError("Data and lines_data attribute are None. Please run from_json() or get_data() & extract_lines_data() first.")

        # If attr != "all": If file_name is None, set it to the same as attr parameter and add .json
        if attr != "all":
            if file_name is None:
                file_name = attr + ".json"
            
            # If file_name isn't None and attr != "all": Check if file_name ends with .json, if not, add it
            else:
                if not file_name.endswith(".json"):
                    file_name += ".json"

        # If attr == "all": If file_name is None, set it to ["data.json", "lines_data.json"]
        else:
            if file_name is None:
                file_name = ["data.json", "lines_data.json"]
            
            # If file_name isn't None and attr == "all": Check if file_name is a list of two strings, if not raise error
            else:
                if not isinstance(file_name, list) or len(file_name) != 2:
                    if not all([isinstance(x, str) for x in file_name]):
                        raise ValueError(f"file_name must be a list of two strings, not {file_name}.")
                
                # Check if file_names end with .json, if not, add it
                for i, fn in enumerate(file_name):
                    if not fn.endswith(".json"):
                        file_name[i] = fn + ".json"

        # If to_dir is None, set it to the current working directory
        if to_dir is None:
            to_dir = Path.cwd()
        
        # Otherwise, check if to_dir exists. If not, create it
        else:
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)

        # If attr is "all", save both data and lines_data attribute, otherwise save the attribute specified by attr parameter
        if attr == "all":
            self.to_json(attr="data", file_name=file_name[0], to_dir=to_dir, overwrite=overwrite)
            self.to_json(attr="lines_data", file_name=file_name[1], to_dir=to_dir, overwrite=overwrite)
        else:
            # Get save_path
            save_path = Path(to_dir).joinpath(file_name)

            # Check if save_path already exists
            if os.path.exists(save_path):
                # If overwrite is True, delete existing json file
                if overwrite:
                    os.remove(save_path)
                # If overwrite is False, raise error
                else:
                    raise FileExistsError(f"File {save_path} already exists. If you want to replace it, set overwrite to True.")

            # Save json file
            if attr == "data":
                self.data.to_json(save_path.as_posix())
            elif attr == "lines_data":
                self.lines_data.to_json(save_path.as_posix())
        

    
    
    def from_json(self, attr: str = "all", file_name = None, from_dir: str = None) -> pd.DataFrame:
        """
        Load data and/or lines_data attribute (DataFrame) from json file.
        :param attr: The attribute containing the data to save as json. Can be "data", "lines_data" or "all". If set to "all", both data and lines_data attributes will be loaded. Default is "all".
        :param file_name: The name of the json file to load. If None, the name of the json file will be the same set to the attr parameter. If attr parameter is set to "all", a list of file names ["data_file", "lines_data_file"] can be passed. Default is None.
        :param from_dir: The directory to load the json file from. If None, the json file will be loaded from the current working directory. Default is None.
        :return: The loaded DataFrame.
        """
        
        # Check if attr is valid
        if attr not in ["data", "lines_data", "all"]:
            raise ValueError(f"attr must be either 'data', 'lines_data' or 'all', not {attr}.")

        # If attr != "all": If file_name is None, set it to the same as attr parameter and add .json
        if attr != "all":
            if file_name is None:
                file_name = attr + ".json"
            
            # If file_name isn't None and attr != "all": Check if file_name ends with .json, if not, add it
            else:
                if not file_name.endswith(".json"):
                    file_name += ".json"

        # If attr == "all": If file_name is None, set it to ["data.json", "lines_data.json"]
        else:
            if file_name is None:
                file_name = ["data.json", "lines_data.json"]
            
            # If file_name isn't None and attr == "all": Check if file_name is a list of two strings, if not raise error
            else:
                if not isinstance(file_name, list) or len(file_name) != 2:
                    if not all([isinstance(x, str) for x in file_name]):
                        raise ValueError(f"file_name must be a list of two strings, not {file_name}.")
                
                # Check if file_names end with .json, if not, add it
                for i, fn in enumerate(file_name):
                    if not fn.endswith(".json"):
                        file_name[i] = fn + ".json"

        # If from_dir is None, set it to the current working directory
        if from_dir is None:
            from_dir = Path.cwd()
        
        # Otherwise, check if from_dir exists. If not, raise error
        else:
            if not os.path.exists(from_dir):
                raise FileNotFoundError(f"Directory {from_dir} does not exist.")
        
        # If attr is "all", load both data and lines_data attribute, otherwise load the attribute specified by attr parameter
        if attr == "all":
            self.from_json(attr="data", file_name=file_name[0], from_dir=from_dir)
            self.from_json(attr="lines_data", file_name=file_name[1], from_dir=from_dir)
        else:
            # Get load_path
            load_path = Path(from_dir).joinpath(file_name)

            # Check if load_path exists. If not, raise error
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"File {load_path} does not exist.")
            
            # Load json file
            if attr == "data":
                self.data = pd.read_json(load_path.as_posix())
            elif attr == "lines_data":
                self.lines_data = pd.read_json(load_path.as_posix())

        # If attr is "all", return both data and lines_data attributes, otherwise return the attribute specified by attr parameter
        if attr == "all":
            return self.data, self.lines_data
        else:
            return getattr(self, attr)