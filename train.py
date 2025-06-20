import os
from tqdm import *
import random
import argparse
import numpy as np
from joblib import dump, load
import torch
import torch.optim as optim
from TRANS_utils import *
from data.Task import *
from models.Model import *
from models.baselines import *

import pathlib
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/sub2vec/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # ddx-on-ehr/
from utils.graph_proc import set_train_test_samples

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help = 'Number of epochs to train.')
parser.add_argument('--lr', type=float, default = 0.001, help = 'learning rate.')
parser.add_argument('--model', type=str, default="TRANS", help = 'Transformer, RETAIN, StageNet, KAME, GCT, DDHGNN, TRANS')
parser.add_argument('--dev', type=int, default = 0, help = 'GPU device id')
parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--dataset', type=str, default = "mimic3", choices=['mimic3', 'mimic4', 'ccae'])
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--pe_dim', type=int, default = 4, help = 'dimensions of spatial encoding')
parser.add_argument('--devm', type=bool, default = False, help = 'develop mode')


fileroot = {
   'mimic3': 'data path of mimic3',
   'mimic4': 'data path of mimic4',
   'ccae': './data/processed_dip.pkl'
}

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print('{}--{}'.format(args.dataset, args.model))
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid if torch.cuda.is_available() else "cpu")

if args.dataset == 'mimic4':
    # Our preprocessed sample dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sample_dataset_path = pathlib.Path(project_root) / 'dataset/preprocessed_data/sample_dataset.pkl'

    if os.path.exists(sample_dataset_path):
        with open(sample_dataset_path, 'rb') as f:
            task_dataset = pickle.load(f)

#    task_dataset = load_dataset(args.dataset, root = fileroot[args.dataset], task_fn=diag_prediction_mimic4_fn, dev= args.devm)
elif args.dataset == 'mimic3':
   task_dataset = load_dataset(args.dataset, root = fileroot[args.dataset], task_fn=diag_prediction_mimic3_fn, dev= args.devm)
else:
    task_dataset = load_dataset(args.dataset, root = fileroot[args.dataset])

# print(task_dataset.input_info)
# print(task_dataset.samples[0]['lab_itemid'])
# exit()

# Overwrite the input_info to fix task_dataset.get_all_tokens()
task_dataset.input_info['lab_itemid']['type'] = str
task_dataset.input_info['lab_itemid']['dim'] = 3

for sample in task_dataset.samples:
    sample['labels'] = sample['labels'].index(1)
task_dataset.input_info['labels']['type'] = str
task_dataset.input_info['labels']['dim'] = 0     

# print(task_dataset.input_info)
# exit()

Tokenizers = get_init_tokenizers(task_dataset, keys=['lab_itemid'])
label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('labels'))

# =========

if args.model == 'Transformer':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    model  = Transformer(Tokenizers,len(task_dataset.get_all_tokens('conditions')),device)

elif args.model == 'RETAIN':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    model  = RETAIN(Tokenizers,len(task_dataset.get_all_tokens('conditions')),device)

elif args.model == 'KAME':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    Tokenizers.update(get_parent_tokenizers(task_dataset))
    model  = KAME(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'StageNet':
    train_loader , val_loader, test_loader = seq_dataloader(task_dataset, batch_size = args.batch_size)
    model  = StageNet(Tokenizers, len(task_dataset.get_all_tokens('conditions')), device)

elif args.model == 'TRANS':
    train_data_path = './logs/train_{}_{}.pkl'.format(args.dataset, args.pe_dim)
    test_data_path = './logs/test_{}_{}.pkl'.format(args.dataset, args.pe_dim)
    
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        train_samples, test_samples = set_train_test_samples(sample_dataset=task_dataset, dev=False, test_size=0.25, split_seed=42, split=True)

        if os.path.exists(train_data_path):
            print('Loading preprocessed data from {}'.format(train_data_path))
            trainset = load(train_data_path)
            print('Loaded preprocessed data with {} samples'.format(len(trainset)))
        else:
            trainset = MMDataset(train_samples, Tokenizers, dim = 128, device = device, trans_dim=args.pe_dim)
            print('Saving preprocessed data to {}'.format(train_data_path))
            dump(trainset,train_data_path)

        if os.path.exists(test_data_path):
            print('Loading preprocessed data from {}'.format(test_data_path))
            testset = load(test_data_path)
            print('Loaded preprocessed data with {} samples'.format(len(testset)))
        else:
            testset = MMDataset(test_samples, Tokenizers, dim = 128, device = device, trans_dim=args.pe_dim)
            print('Saving preprocessed data to {}'.format(test_data_path))
            dump(testset,test_data_path)
    else:
        print('Loading preprocessed data from {} and {}'.format(train_data_path, test_data_path))
        trainset = load(train_data_path)
        testset = load(test_data_path)
        print('Loaded preprocessed data with {} train samples and {} test samples'.format(len(trainset), len(testset)))

        # dump(trainset[:1000], './logs/train_{}_{}_1000.pkl'.format(args.dataset, args.pe_dim))
        # dump(testset[:250], './logs/test_{}_{}_250.pkl'.format(args.dataset, args.pe_dim))

    # trainset, validset, testset = split_dataset(mdataset)
    # train_loader , val_loader, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)
    train_loader, test_loader = mm_dataloader(trainset, testset, batch_size=args.batch_size)

    # print('task_dataset.get_all_tokens("labels"):', task_dataset.get_all_tokens('labels'))
    
    model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('labels')),
                    device,graph_meta=graph_meta, pe=args.pe_dim)

ckptpath = './logs/trained_{}_{}.ckpt'.format(args.model, args.dataset)    ###
optimizer =torch.optim.AdamW(model.parameters(), lr = args.lr)
best = 12345
pbar = tqdm(range(args.epochs))

if os.path.exists(ckptpath):
    print(f"Checkpoint {ckptpath} exists, skipping training.")
else:
    print(f"Checkpoint {ckptpath} does not exist, starting training.")
    for epoch in pbar:
        model = model.to(device)

        train_loss = train(train_loader, model, label_tokenizer, optimizer, device)
        # val_loss = valid(val_loader, model, label_tokenizer, device)

        # pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.2f} - valid loss: {val_loss:.2f}")
        # if val_loss<best:
        #     torch.save(model.state_dict(), ckptpath)

        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} - train loss: {train_loss:.2f}")
        if train_loss<best:
            torch.save(model.state_dict(), ckptpath)

#for limited gpu memory
if args.model == 'TRANS':
    del model
    torch.cuda.empty_cache()
    import gc

    gc.collect()
    device = torch.device('cpu')
    model = TRANS(Tokenizers, 128, len(task_dataset.get_all_tokens('labels')),
                device, graph_meta=graph_meta, pe=args.pe_dim)
best_model = torch.load(ckptpath, weights_only=True)    # weights_only=True : 只載入權重，避免安全警告
model.load_state_dict(best_model, strict=False)     # strict=False : 忽略多餘或缺少的權重
model = model.to(device)

y_t_all, y_p_all = [], []
y_true, y_pred, y_prob = test(test_loader, model, label_tokenizer)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

labels = task_dataset.get_all_tokens('labels')

report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Save reports
result_path = pathlib.Path("./result/{}_{}_{}".format(args.model, args.dataset, args.pe_dim))
result_path.mkdir(parents=True, exist_ok=True)
with open(result_path / "classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(result_path / "confusion_matrix.png")

# print(code_level(y_true, y_prob))
# print(visit_level(y_true, y_prob))
