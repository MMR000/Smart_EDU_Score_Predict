from typing import List
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, cat_dims: List[int], emb_dim: int = 16):
        super().__init__()
        self.embs = nn.ModuleList()
        for d in cat_dims:
            # +1 because 0 is unknown
            ed = min(emb_dim, int(round(1.6 * (d ** 0.56))) + 1)
            self.embs.append(nn.Embedding(d + 1, ed))
        self.out_dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, x_cat):
        outs = []
        for j, e in enumerate(self.embs):
            outs.append(e(x_cat[:, j]))
        return torch.cat(outs, dim=1) if outs else None

class MLPEmb(nn.Module):
    def __init__(self, n_num: int, cat_dims: List[int], hidden: List[int], dropout: float, out_dim: int):
        super().__init__()
        self.emb = Embeddings(cat_dims, emb_dim=24)
        in_dim = n_num + (self.emb.out_dim if cat_dims else 0)
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        parts = [x_num]
        if x_cat is not None and x_cat.shape[1] > 0:
            parts.append(self.emb(x_cat))
        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.net(x)

class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, 1) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        xw = x @ self.w  # (B,1)
        return x0 * xw + self.b + x

class DeepCross(nn.Module):
    def __init__(self, n_num: int, cat_dims: List[int], deep_hidden: List[int], cross_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.emb = Embeddings(cat_dims, emb_dim=24)
        self.in_dim = n_num + (self.emb.out_dim if cat_dims else 0)
        self.cross = nn.ModuleList([CrossLayer(self.in_dim) for _ in range(cross_layers)])

        d = self.in_dim
        deep = []
        for h in deep_hidden:
            deep += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.deep = nn.Sequential(*deep) if deep else nn.Identity()

        deep_out_dim = (deep_hidden[-1] if deep_hidden else self.in_dim)
        self.fc = nn.Linear(self.in_dim + deep_out_dim, out_dim)

    def forward(self, x_num, x_cat):
        parts = [x_num]
        if x_cat is not None and x_cat.shape[1] > 0:
            parts.append(self.emb(x_cat))
        x0 = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        deep = self.deep(x0)
        out = torch.cat([x, deep], dim=1)
        return self.fc(out)

class MultiheadBlock(nn.Module):
    def __init__(self, d, n_heads, dropout, ffn_factor=2.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, int(d * ffn_factor)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d * ffn_factor), d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(h))
        h = self.ff(x)
        x = self.ln2(x + self.drop(h))
        return x

class FTTransformer(nn.Module):
    # Feature-token transformer (FT-Transformer style)
    def __init__(self, n_num: int, cat_dims: List[int], d_token: int, n_blocks: int, n_heads: int, dropout: float, ffn_factor: float, out_dim: int):
        super().__init__()
        self.n_num = n_num
        self.cat_dims = cat_dims

        self.num_w = nn.Parameter(torch.randn(n_num, d_token) * 0.01) if n_num > 0 else None
        self.num_b = nn.Parameter(torch.zeros(n_num, d_token)) if n_num > 0 else None

        self.cat_embs = nn.ModuleList([nn.Embedding(d + 1, d_token) for d in cat_dims])

        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))

        self.blocks = nn.ModuleList([MultiheadBlock(d_token, n_heads, dropout, ffn_factor) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, out_dim))

    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        tokens = []
        if self.n_num > 0:
            xn = x_num.unsqueeze(-1)  # (B, n_num, 1)
            tnum = xn * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)
            tokens.append(tnum)
        if x_cat is not None and x_cat.shape[1] > 0:
            tcat = [e(x_cat[:, j]).unsqueeze(1) for j, e in enumerate(self.cat_embs)]
            tokens.append(torch.cat(tcat, dim=1))
        x = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, 0])

class AutoInt(nn.Module):
    def __init__(self, n_num: int, cat_dims: List[int], d_token: int, n_blocks: int, n_heads: int, dropout: float, out_dim: int):
        super().__init__()
        self.n_num = n_num
        self.num_proj = nn.Linear(n_num, n_num * d_token) if n_num > 0 else None
        self.cat_embs = nn.ModuleList([nn.Embedding(d + 1, d_token) for d in cat_dims])
        self.blocks = nn.ModuleList([MultiheadBlock(d_token, n_heads, dropout, ffn_factor=2.0) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, out_dim))

    def forward(self, x_num, x_cat):
        B = x_num.shape[0]
        tokens = []
        if self.n_num > 0:
            t = self.num_proj(x_num).view(B, self.n_num, -1)
            tokens.append(t)
        if x_cat is not None and x_cat.shape[1] > 0:
            tcat = [e(x_cat[:, j]).unsqueeze(1) for j, e in enumerate(self.cat_embs)]
            tokens.append(torch.cat(tcat, dim=1))
        x = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        return self.out(x)
