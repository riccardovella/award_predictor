import argparse
import sys

from pathlib import Path

from datetime import datetime

import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import Small_Graphs_Dataset

from sklearn.model_selection import train_test_split

from model import BinaryClassificationMLP, MixedNetwork
from model import build_MLP_from_save

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from train_model import train
from logger import Logger
from test_model import test_and_plot

def load_np_dataset(path):
    data = np.load(path)
    return data['X'], data['y']

def load_gm_dataset(path): 
    X, y = load_np_dataset(path)
    X = X[:, 0:5]
    X = (X - X.min()) / (X.max() - X.min()) # features normalization
    return X, y

def load_tx_dataset(path):
    return load_np_dataset(path)

def main(args):
    # find running device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load and prepare dataset
    if args.gm_data_path:
        X, y = load_gm_dataset(args.gm_data_path)
    else:
        X, y = load_tx_dataset(args.tx_data_path)

    if args.m:
        X_, y_ = load_np_dataset(args.tx_data_path)
        if not np.array_equal(y, y_):
            raise Exception("Trying to concat inconsistent data.")
        X = np.concatenate([X, X_], axis=1)

    in_size = len(X[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    BATCH_SIZE = 64  # not defined in args, can be changed

    train_dataset = Small_Graphs_Dataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    test_dataset = Small_Graphs_Dataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # prepare model
    if args.m:
        model_graph = build_MLP_from_save(args.gm_model_path, device)
        model_text = build_MLP_from_save(args.tx_model_path, device)

        model_graph.load(args.gm_model_path, device)
        model_text.load(args.tx_model_path, device)

        model = MixedNetwork(model_graph, model_text, args.hidden)
    else:
        model = BinaryClassificationMLP(in_size, args.hidden)

    model.to(device)

    # The dataset is unbalanced, define a weight for positive classes
    weight = torch.tensor((y==0.).sum()/y.sum(), dtype=torch.float32)

    loss_fn = BCEWithLogitsLoss(pos_weight=weight)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # create out directory 
    out_dir = Path(args.outdir)
    if not out_dir.exists():
        out_dir.mkdir()
    if args.unique_name: 
        dt = datetime.now()
        args.name += "_" + f"{dt:%Y-%m-%d_%H:%M:%S}.{dt.microsecond // 1000:03d}"
    out_dir = out_dir / args.name
    out_dir.mkdir()

    # create plot directory
    plots_dir = out_dir / "plots"
    plots_dir.mkdir()

    logger = Logger()

    train(train_dataloader, test_dataloader, model, loss_fn, optimizer, 
          args.epochs, device, logger, out_dir)
    
    # save training status plots
    logger.save_plot(f"{plots_dir}/train_test_loss.png",
                            key=["train_loss", "test_loss"], y_label="Loss")
    logger.save_plot(f"{plots_dir}/accuracy_f1.png",
                        key=["accuracy", "f1-score"], y_label="Score")
    
    # load best model
    model.load(str(out_dir / "best_f1.pt"), device)
    
    test_and_plot(plots_dir, model, loss_fn, test_dataloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Research Award Predictor",
        description='Trains and tests the machine learning models described in the research' + \
        ' and thesis project "Predicting Award Winning Research Papers at Publication Time"'
    )

    parser.add_argument("outdir", type=Path,
                        help="The directory where models and results are saved")
    parser.add_argument("--gm_data_path", type=Path, required='--tx_data_path' not in sys.argv,
                        help="The path of the graph measures dataset")
    parser.add_argument("--tx_data_path", type=Path, required='--gm_data_path' not in sys.argv,
                        help="The path of the text features dataset")
    parser.add_argument("-hd", "--hidden", type=int, required=True,
                        help="The number of nodes in the hidden layer")
    parser.add_argument("-l", "--lr", type=float, required=True,
                        help="The learning rate")
    parser.add_argument("-e", "--epochs", type=int, required=True,
                        help="The number of epochs")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="The random seed")
    parser.add_argument("-n", "--name", type=str, default="model_results", 
                        help="The name of the model")
    parser.add_argument("-u", "--unique_name", action="store_true", 
                        help="Makes the name of the model unique by adding the date")
    
    # MIXED MODE ARGUMENTS
    parser.add_argument("-m", action='store_true',
                        help="Instantiates a mixed model")
    # these are only required if --mixed_mode is given
    parser.add_argument("--gm_model_path", type=Path, required='-m' in sys.argv,
                        help="The path of the graph measures trained model")
    parser.add_argument("--tx_model_path", type=Path, required='-m' in sys.argv,
                        help="The path of the text features trained model")

    args = parser.parse_args()

    if not args.m and args.gm_data_path and args.tx_data_path:
        raise ValueError("Cannot use both --gm_data_path and --tx_data_path when -m (mixed mode) is off")

    main(args)