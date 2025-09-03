import os
import argparse
import numpy as np
import pathlib
import sys

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/sub2vec/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # ddx-on-ehr/
from utils.PCA_analysis import plot_PCA_3D

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="TRANS", help = 'Transformer, RETAIN, StageNet, KAME, GCT, DDHGNN, TRANS')
parser.add_argument('--dataset', type=str, default = "mimic3", choices=['mimic3', 'mimic4', 'ccae'])
parser.add_argument('--pe_dim', type=int, default = 4, help = 'dimensions of spatial encoding')
parser.add_argument('--time_slice_pct', type=float, default=1.0, help='Percentage of time for slicing test samples')

args = parser.parse_args()

print('{}--{}'.format(args.dataset, args.model))

result_path = pathlib.Path("./result/{}_{}_{}_{}".format(args.model, args.dataset, args.pe_dim, args.time_slice_pct))
result_path.mkdir(parents=True, exist_ok=True)

data = np.load(result_path / "test_outputs.npy", allow_pickle=True).item()
X = data['combined_feature']
y_true = data['y_true']
y_pred = data['y_pred']
y_prob = data['y_prob']

plot_PCA_3D(X, y_true, y_pred, y_prob, save_root=result_path)