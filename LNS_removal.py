import numpy as np
import random
import torch
import os
import json
import warnings

import onlinehd
from sklearn.preprocessing import Normalizer
from torch.utils.data import TensorDataset, random_split

warnings.filterwarnings("ignore")

def test(model, C_prune, X, Y):
    model.model = torch.tensor(C_prune, dtype=torch.float32)
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32))
    return (preds == torch.tensor(Y)).float().mean()

def lns_column_pruning(
    model,
    C,
    X_val_encoded,
    Y_val,
    target_columns,
    folder,
    iterations=8000,
    ruin_fraction=0.05,
    recreate_prob=0.3,
    lambda_=2.0,
    acc_threshold=0.74,
    patience=8000
):
    D = len(target_columns)
    eval_cache = {}

    def test_pruning(mask):
        key = tuple(mask)
        if key in eval_cache:
            return eval_cache[key]

        C_prune = C.clone()
        to_zero = [target_columns[i] for i, bit in enumerate(mask) if bit == 0]
        if to_zero:
            C_prune[:, to_zero] = 0

        model.model = torch.tensor(C_prune, dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_val_encoded, encoded=True)
        acc = (preds == Y_val).float().mean().item()

        score = acc + lambda_ * (1 - mask.sum() / D) if acc >= acc_threshold else float('-inf')
        eval_cache[key] = (score, acc, C_prune)
        return score, acc, C_prune

    current_mask = (np.random.rand(D) < 0.8).astype(np.uint8)
    best_score, best_acc, best_C_pruned = test_pruning(current_mask)
    best_mask = current_mask.copy()
    count = 0

    for it in range(1, iterations + 1):
        ruined_mask = current_mask.copy()
        ruin_indices = np.random.choice(D, int(ruin_fraction * D), replace=False)
        ruined_mask[ruin_indices] = 0
        for idx in ruin_indices:
            if random.random() < recreate_prob:
                ruined_mask[idx] = 1

        new_score, new_acc, new_C_pruned = test_pruning(ruined_mask)

        if new_score > best_score:
            best_score = new_score
            best_acc = new_acc
            best_C_pruned = new_C_pruned
            best_mask = ruined_mask.copy()
            current_mask = ruined_mask.copy()
            count = 0
        else:
            count += 1

        if count >= patience:
            break

    kept_columns = [col for col, bit in zip(target_columns, best_mask) if bit]
    return best_C_pruned, best_acc, kept_columns

def random_column_removal_by_rate(model, C, keep_rate):
    D = C.shape[1]
    num_keep = int(D * keep_rate)
    kept_indices = sorted(random.sample(range(D), num_keep))

    C_random = C.clone()
    all_indices = set(range(D))
    remove_indices = list(all_indices - set(kept_indices))
    C_random[:, remove_indices] = 0

    return C_random

def abs_column_summation_removal(C, keep_rate):
    col_sums = torch.sum(torch.abs(C), dim=0)
    sorted_indices = torch.argsort(col_sums)
    D = C.shape[1]
    num_keep = int(D * keep_rate)
    keep_indices = sorted_indices[-num_keep:].tolist()
    C_abs = C.clone()
    remove_indices = list(set(range(D)) - set(keep_indices))
    C_abs[:, remove_indices] = 0
    return C_abs

def load_data(folder):
    print(f"+++ Loading {folder} +++")

    x = np.load(f'./data/{folder}/X_train.npy')
    y = np.load(f'./data/{folder}/y_train.npy')
    x_test = np.load(f'./data/{folder}/X_test.npy')
    y_test = np.load(f'./data/{folder}/y_test.npy')

    x = x.reshape(len(x), -1)
    x_test = x_test.reshape(len(x_test), -1)
    scaler = Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y.flatten(), dtype=torch.long),
        torch.tensor(y_test.flatten(), dtype=torch.long),
        folder
    )

def main():
    folders = ['ucihar', 'isolet', 'gtsrb', 'fashion_mnist', 'mnist']
    for folder in folders:
        os.makedirs(f'./model_{folder}', exist_ok=True)

        x, x_test, y, y_test, name = load_data(folder)
        if folder == 'ucihar':
            y = y - 1
            y_test = y_test - 1

        classes = y.unique().numel()
        features = x.size(1)
        D = 10000

        dataset = TensorDataset(x, y)
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))
        x_train, y_train = train_set[:][0], train_set[:][1]
        x_val, y_val = val_set[:][0], val_set[:][1]

        model = onlinehd.OnlineHD(classes, features, dim=D)
        model = model.fit(x_train, y_train, bootstrap=1.0, lr=0.035, epochs=200)

        # Save the original model
        torch.save(model, f'./model_{folder}/model.pt')
        C_bipolar = torch.sign(model.model.clone())
        model.model = C_bipolar
        initial_test_acc = (model(x_test) == y_test).float().mean().item()

        C = model.model.clone()
        x_val_encoded = model.encode(x_val)
        target_columns = list(range(D))

        pruned_C_lns, val_acc_lns, kept_columns_lns = lns_column_pruning(
            model=model,
            C=C.clone(),
            X_val_encoded=x_val_encoded,
            Y_val=y_val,
            target_columns=target_columns,
            folder=folder,
            iterations=20000,
            ruin_fraction=0.1,
            recreate_prob=0.2,
            lambda_=1.0,
            acc_threshold=0.95 * initial_test_acc,
            patience=3000
        )

        # Save pruned model and kept column indices
        pruned_model_path = f'./model_{folder}/pruned_model.pt'
        kept_columns_path = f'./model_{folder}/kept_columns.pt'
        torch.save(torch.tensor(pruned_C_lns, dtype=torch.float32), pruned_model_path)
        torch.save(kept_columns_lns, kept_columns_path)

        keep_rate = len(kept_columns_lns) / D

        model.model = torch.tensor(pruned_C_lns, dtype=torch.float32)
        final_test_acc_lns = (model(x_test) == y_test).float().mean().item()

        pruned_C_random = random_column_removal_by_rate(model, C.clone(), keep_rate)
        model.model = torch.tensor(pruned_C_random, dtype=torch.float32)
        final_test_acc_random = (model(x_test) == y_test).float().mean().item()

        pruned_C_abs = abs_column_summation_removal(C.clone(), keep_rate)
        model.model = torch.tensor(pruned_C_abs, dtype=torch.float32)
        final_test_acc_abs = (model(x_test) == y_test).float().mean().item()

if __name__ == "__main__":
    main()

