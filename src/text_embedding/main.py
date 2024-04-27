import argparse
from pathlib import Path

import numpy as np

from features_extractor import extract_features_from_small_graphs

def main(args):
    X, y = extract_features_from_small_graphs(args.small_graphs_path)

    np.savez(args.text_features_path, X=X, y=y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Text Embedding Builder",
        description="Builds the text embedding file from the small graphs dataset"
    )

    parser.add_argument("small_graphs_path", type=Path,
                        help="The path for the small graph dataset")
    parser.add_argument("text_features_path", type=Path,
                        help="The output path where to save the resulting dataset")
    
    args = parser.parse_args()

    main(args)