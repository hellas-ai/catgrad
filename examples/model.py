# The model saved by this run with default arguments
# can be loaded by the example apps.

import argparse
from safetensors.torch import save_model

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim=8, exp=2):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim * exp, bias=True)
        self.lin2 = nn.Linear(dim * exp, dim, bias=True)
        self.act = nn.Tanh()
        # self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim):
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.size()
        return self.proj(x)

class Layer(nn.Module):
    def __init__(self, dim=8, exp=2):
        super().__init__()
        self.mlp = MLP(dim, exp)

    def forward(self, x):
        res = x
        x = self.mlp(x)
        return x + res

class Model(nn.Module):
    def __init__(self, vocab_size, layers, dim, exp):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.prenorm = nn.LayerNorm(dim)
        self.layers = nn.Sequential(*[Layer(dim, exp) for _ in range(layers)])
        self.postnorm = nn.LayerNorm(dim)

    def forward(self, x):
        # x = self.token_embeddings(x)
        x = self.prenorm(x)
        x = self.layers(x)
        x = self.postnorm(x)
        x = F.softmax(x, dim=-1)
        return x


def main(args):
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float16

    torch.manual_seed(args.seed)

    torch.set_printoptions(precision=6)
    x = torch.full((args.dim,), 1.0)
    print(x)

    model = Model(args.vocab_size, args.layers, args.dim, args.exp)

    print(model)
    y = model(x)

    print(y)

    save_model(model, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", type=str, default="model.safetensors")
    parser.add_argument("--layers", "-l", type=int, default=4)
    parser.add_argument("--vocab-size", "-v", type=int, default=128)
    parser.add_argument("--dim", "-d", type=int, default=8)
    parser.add_argument("--exp", "-e", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", "-s", type=int, default=2)
    args = parser.parse_args()

    main(args)
