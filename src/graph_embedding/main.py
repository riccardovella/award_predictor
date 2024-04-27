import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from networkx import DiGraph

from tqdm.auto import tqdm
tqdm.pandas()

from graph_measures import get_graph_measures

def create_dataset(df, approximate=False):
    def process_one(row):
        G = DiGraph()
        G.add_nodes_from(row['nodes'])
        G.add_edges_from(row['edges'])

        x = get_graph_measures(G, approximate)
        x.append(row['year'] - 1999) # adds timestamp
        y = 1 if row['has_award'] else 0

        return x, y

    data = df.progress_apply(
        lambda x: process_one(x), axis=1, result_type="expand")

    X = np.array(data[0].values.tolist())
    y = data[1].to_numpy()

    return X, y

def main(args):
    df = pd.read_parquet(args.small_graphs_path, engine='fastparquet')

    X, y = create_dataset(df, approximate=False)

    np.savez(args.graph_measures_path, X=X, y=y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Graph Embedding Builder",
        description="Builds the graph embedding file from the small graphs dataset"
    )

    parser.add_argument("small_graphs_path", type=Path,
                        help="The path for the small graph dataset")
    parser.add_argument("graph_measures_path", type=Path,
                        help="The output path where to save the resulting dataset")
    
    args = parser.parse_args()

    main(args)