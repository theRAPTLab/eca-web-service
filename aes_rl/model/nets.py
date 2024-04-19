import torch.nn as nn
import torch
import dataclasses


class QNetwork(nn.Module):
    def __init__(self, args: dataclasses) -> None:
        super(QNetwork, self).__init__()
        self.args = args
        self.q1 = nn.Linear(self.args.state_dim, self.args.units)
        self.q2 = nn.Linear(self.args.units, self.args.units)
        self.q3 = nn.Linear(self.args.units, self.args.action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q = torch.tanh(self.q1(state))
        q = torch.tanh(self.q2(q))
        q = self.q3(q)
        return q


class BCNetwork(nn.Module):
    def __init__(self, args: dataclasses):
        super(BCNetwork, self).__init__()
        self.args = args

        self.i1 = nn.Linear(self.args.state_dim, self.args.units)
        self.i2 = nn.Linear(self.args.units, self.args.units)
        self.i3 = nn.Linear(self.args.units, self.args.action_dim)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        i = torch.tanh(self.i1(state))
        i = torch.tanh(self.i2(i))
        i = torch.tanh(self.i3(i))
        i_log_softmax = nn.functional.softmax(i, dim=1)

        return i_log_softmax, i
