import argparse
import os

import pandas as pd

from tqdm import tqdm
from general_utils import create_directory

if __name__ == "__main__":
    print(
        "\n--------------------\nGenerating Train and Val CSVs!\n--------------------\n"
    )

    parser = argparse.ArgumentParser(description="Argparse to create CSVs")
    parser.add_argument(
        "--input", "-I", type=str, help="Input directory", required=True
    )
    parser.add_argument(
        "--output", "-O", type=str, help="Output directory", required=True
    )
    args = parser.parse_args()
    input_directory = args.input
    output_directory = args.output

    for folder_name in tqdm(os.listdir(input_directory)):
        if folder_name == "test" or folder_name.startswith(".DS"):
            continue

        input_path = os.path.join(input_directory, folder_name, "3triples")

        output_path = os.path.join(output_directory, folder_name)
        create_directory(directory=output_path, print_flag=False)

        csv_path = os.path.join(output_path, "3triples.csv")
        dataframe = pd.DataFrame()

        for file_name in os.listdir(input_path):
            file_path = os.path.join(input_path, file_name)
            df = pd.read_csv(file_path)
            dataframe = pd.concat([dataframe, df], axis=0)
        dataframe.reset_index(drop=True)
        dataframe = dataframe.to_csv(csv_path, index=False)

    print(
        "\n--------------------\nGenerated Train and Val CSVs!\n--------------------\n"
    )
