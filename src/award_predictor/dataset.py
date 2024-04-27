from torch.utils.data import Dataset
from torch import tensor, float32

class Small_Graphs_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = tensor(X, dtype=float32)
        self.y = tensor(y, dtype=float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]