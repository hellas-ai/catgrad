# The model saved by this run with default arguments
# can be loaded by the example apps.

import math
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
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.lin1(x)
        x = self.tanh(x)
        x = self.lin2(x)
        x = self.gelu(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # B, T, D = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        if False:  # these are equivalent
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            attn = q @ k.mT
            attn = attn / math.sqrt(k.size(-1))
            attn = torch.nn.functional.softmax(attn, dim=-1)
            attn = attn @ v
        return self.proj(attn)


class Layer(nn.Module):
    def __init__(self, dim=8, exp=2):
        super().__init__()
        self.prenorm = nn.RMSNorm(dim)
        # Init to something other than the default 1
        # to see that catgrad is using the norm weights
        nn.init.constant_(self.prenorm.weight, 0.43)
        self.attention = Attention(dim)
        self.postnorm = nn.RMSNorm(dim)
        self.mlp = MLP(dim, exp)

    def forward(self, x):
        res = x
        x = self.prenorm(x)
        x = self.attention(x)
        x = self.postnorm(x)
        x = self.mlp(x)
        return x + res


class Model(nn.Module):
    def __init__(self, vocab_size, max_seq_len, layers, dim, exp):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_seq_len, dim)
        self.prenorm = nn.LayerNorm(dim)
        self.layers = nn.Sequential(*[Layer(dim, exp) for _ in range(layers)])
        self.postnorm = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, seq_len = x.shape
        te = self.token_embeddings(x)
        pos = torch.arange(0, seq_len)
        pe = self.position_embeddings(pos)
        x = te + pe
        x = self.prenorm(x)
        x = self.layers(x)
        x = self.postnorm(x)
        x = F.softmax(x, dim=-1)
        return x


def main(args):
    args.dtype = torch.float32 if args.dtype == "float32" else torch.float16

    torch.manual_seed(args.seed)

    torch.set_printoptions(precision=6)
    x = torch.full((args.batches, args.tokens), args.fill)
    print(x)

    model = Model(args.vocab_size, args.max_seq_len, args.layers, args.dim, args.exp)

    # print(model)
    y = model(x)

    print(y)

    save_model(model, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-p", type=str, default="model.safetensors")
    parser.add_argument("--batches", "-b", type=int, default=1)
    parser.add_argument("--tokens", "-t", type=int, default=1)
    parser.add_argument("--max-seq-len", "-m", type=int, default=16)
    parser.add_argument("--fill", "-f", type=int, default=1)
    parser.add_argument("--layers", "-l", type=int, default=4)
    parser.add_argument("--vocab-size", "-v", type=int, default=128)
    parser.add_argument("--dim", "-d", type=int, default=8)
    parser.add_argument("--exp", "-e", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--seed", "-s", type=int, default=2)
    args = parser.parse_args()

    main(args)
