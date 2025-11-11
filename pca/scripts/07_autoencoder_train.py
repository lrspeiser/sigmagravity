#!/usr/bin/env python3
import argparse, os, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class CurveDataset(Dataset):
    def __init__(self, X, W=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.W = None if W is None else torch.as_tensor(W, dtype=torch.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        if self.W is None:
            return self.X[idx], torch.ones_like(self.X[idx])
        return self.X[idx], self.W[idx]

class AE(nn.Module):
    def __init__(self, D, latent=3):
        super().__init__()
        hid = max(16, D//2)
        self.enc = nn.Sequential(nn.Linear(D, hid), nn.ReLU(), nn.Linear(hid, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hid), nn.ReLU(), nn.Linear(hid, D))
    def forward(self, x):
        z = self.enc(x)
        y = self.dec(z)
        return y, z

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_npz", required=True, help="Use curve-only features (X, W).")
    ap.add_argument("--latent_dim", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.features_npz, allow_pickle=True)
    X = data["X"]; W = data["W"]
    D = X.shape[1]
    ds = CurveDataset(X, W)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(D, latent=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for xb, wb in dl:
            xb = xb.to(device); wb = wb.to(device)
            yb, zb = model(xb)
            # weighted MSE (per-feature weights)
            loss = torch.mean(wb * (yb - xb)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        avg = total / len(ds)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} | loss {avg:.6f}")
    # Save model and latent codes
    model.eval()
    with torch.no_grad():
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        _, Z = model(X_t)
        Z = Z.cpu().numpy()
    np.savez_compressed(os.path.join(args.out_dir, "autoencoder_latents.npz"), Z=Z)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "autoencoder.pt"))
    print("[OK] Saved AE model and latents.")

if __name__ == "__main__":
    main()
