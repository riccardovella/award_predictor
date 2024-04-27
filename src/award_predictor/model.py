import torch
from torch import nn

def build_MLP_from_save(path, device):
    data = torch.load(path)

    in_size = data["in_size"]
    hid_size = data["hidden_size"]

    model = BinaryClassificationMLP(in_size, hid_size)

    model.load(path, device)

    return model

class BinaryClassificationMLP(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        
        self.hidden = None
        if hidden_size > 0:
            self.hidden = nn.Linear(in_size, hidden_size)
            in_size = hidden_size
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.hidden is not None:
            x = self.relu(self.hidden(x))

        logits = self.output(x)
        x = self.sigmoid(logits)
        return x, logits

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            pred, _ = self.__call__(X)

            y_pred = torch.round(pred).detach().cpu().numpy()
            y_probs = pred.detach().cpu().numpy()

            return y_pred, y_probs

    def load(self, path, device=None):
        if device is not None:
            data = torch.load(path)
        else:
            data = torch.load(path, map_location=torch.device(device))

        self.load_state_dict(data["state_dict"])

    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                    "in_size": self.in_size,
                    "hidden_size": self.hidden_size,
                    }, path)


class MixedNetwork(nn.Module):
    def __init__(self, graph_net, text_net, hidden_size):
        super().__init__()

        self.g_in_size = graph_net.in_size
        self.t_in_size = text_net.in_size

        self.graph_net = graph_net
        self.text_net = text_net

        self.graph_net.requires_grad_(False)
        self.text_net.requires_grad_(False)
        self.graph_net.eval()
        self.text_net.eval()

        in_size = 2
        if hidden_size > 0:
            self.hidden = nn.Linear(in_size, hidden_size)
            in_size = hidden_size

        self.relu = nn.ReLU()

        self.output = nn.Linear(in_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_g, x_t = torch.split(x, [self.g_in_size, self.t_in_size], dim=1)

        out_g, _ = self.graph_net(x_g)
        out_t, _ = self.text_net(x_t)
        x = torch.cat((out_g, out_t), dim=1)

        x = self.relu(self.hidden(x))
        # x = self.hidden(x)

        logits = self.output(x)
        x = self.sigmoid(logits)
        
        return x, logits

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            pred, _ = self.__call__(X)

            y_pred = torch.round(pred).detach().cpu().numpy()
            y_probs = pred.detach().cpu().numpy()

            return y_pred, y_probs

    def load(self, path, device=None):
        if device is None:
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(
                torch.load(path, map_location=torch.device(device)))

    def save(self, path):
        torch.save(self.state_dict(), path)

