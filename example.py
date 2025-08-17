import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from torch.utils.data import TensorDataset, random_split
import math
from fxpmath import Fxp
import argparse

# ==== Error Injection ====
def inject_error(x, prob):
    x_c = x.clone()
    flip_mask = torch.bernoulli(torch.full(x_c.shape, prob)).bool()
    x_c[flip_mask] *= -1
    return x_c, flip_mask

def tmr_vote(cw1, cw2, cw3, cw4, cw5):
    stacked = torch.stack([cw1, cw2, cw3, cw4, cw5], dim=0)
    vote = torch.mode(stacked, dim=0).values
    return vote


def inject_error_scalar(x, p):
    # Convert scalar to fixed-point binary
    fxp = Fxp(x.item(), True, 32, 16)
    bin_str = fxp.bin()  # e.g., '010101010101...'
    bit_array = np.array([int(b) for b in bin_str], dtype=np.int8)

    # Generate bit-flip mask
    flip_mask = np.random.rand(32) < p
    bit_array[flip_mask] ^= 1  # Flip bits

    # Reconstruct binary string and convert back
    flipped_bin_str = '0b' + ''.join(map(str, bit_array))
    return Fxp(flipped_bin_str, True, 32, 16).get_val()
    
    
def inject_error_C(x, er_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x_err = copy.deepcopy(x)
    total_elements = x_err.size
    indices = np.random.choice(total_elements, int(er_rate * total_elements), replace=False)
    error_map = np.zeros_like(x_err, dtype=bool).flatten()
    error_map[indices] = True
    error_map = error_map.reshape(x.shape)
    x_err[error_map] = -x_err[error_map]
    return x_err

def inject_error_C_float(x, p):
    # Convert array to fixed-point binary
    fxp = Fxp(x, True, 32, 16)
    bin_strs = fxp.bin()  # List of binary strings
    bit_array = np.array([[int(b) for b in s] for s in bin_strs], dtype=np.int8)

    # Generate bit-flip mask
    flip_mask = np.random.rand(*bit_array.shape) < p
    bit_array[flip_mask] ^= 1  # Flip bits

    # Reconstruct binary strings and convert back
    flipped_bin_strs = ['0b' + ''.join(map(str, row)) for row in bit_array]
    return Fxp(flipped_bin_strs, True, 32, 16).get_val().reshape(len(x))

def protect_vector_float(x, p, N=3):
    injected = np.array([inject_error_C_float(x, p) for _ in range(N)])
    return np.median(injected, axis=0)
        
def protect_vector(x, p, N=3):
    injected = np.array([inject_error_C(x, p) for _ in range(N)])
    return np.median(injected, axis=0)

def protect_scalar(x, p, N=3):
    injected = np.array([inject_error_scalar(x, p) for _ in range(N)])
    return np.median(injected)


# ==== Cosine Similarity ====
def cos_sim_custom(x, y):
    denom = np.linalg.norm((1 + x) / 2) * np.linalg.norm((1 + y) / 2)
    return np.abs(np.dot(x, y)) / (denom if denom != 0 else 1)
    
def cos_sim(x, y):
    return np.abs(np.dot(x, y)) / 10000

def compute_cosine_refs(C_c, sim_func):
    ref = []
    for i in range(C_c.shape[1]):
        vec = C_c[:, i]
        rolled = np.roll(vec, 1)
        sim = sim_func(vec, rolled)
        ref.append(sim)
    return ref

def detect_errors(C_c_err, ref, sim_func):
    detected_err = []
    for i in range(C_c_err.shape[1]):
        vec = C_c_err[:, i]
        rolled = np.roll(vec, 1)
        sim = sim_func(vec, rolled)
        if np.abs(sim - ref[i])>5e-3:
            detected_err.append(i)
    return detected_err


# ==== Clustering ====
def cluster_critical_with_cosine(C_c_sorted):
    column_vectors = C_c_sorted.T
    cos_sim_matrix = cosine_similarity(column_vectors)
    cos_dist_matrix = 1 - cos_sim_matrix

    clustering = AgglomerativeClustering(n_clusters=250, affinity='precomputed', linkage='average')
    clustering.fit(cos_dist_matrix)

    cluster_dict = defaultdict(list)
    for idx, label in enumerate(clustering.labels_):
        cluster_dict[label].append(idx)

    col_consensus_list, row_consensus_list, Indices = [], [], []

    for cluster_indices in cluster_dict.values():
        cluster = np.array(cluster_indices)
        Indices.append(cluster)
        cluster_matrix = C_c_sorted[:, cluster]

        signs_sum_col = np.sum(cluster_matrix, axis=1)
        col_consensus = np.full(C_c_sorted.shape[0], np.nan)
        col_consensus[signs_sum_col >= 0.6 * cluster_matrix.shape[1]] = 1
        col_consensus[signs_sum_col <= -0.6 * cluster_matrix.shape[1]] = -1
        col_consensus_list.append(col_consensus)

        signs_sum_row = np.sum(cluster_matrix, axis=0)
        row_consensus = np.full(cluster_matrix.shape[1], np.nan)
        row_consensus[signs_sum_row >= 0.65 * cluster_matrix.shape[0]] = 1
        row_consensus[signs_sum_row <= -0.65 * cluster_matrix.shape[0]] = -1
        row_consensus_list.append(row_consensus)

    return Indices, row_consensus_list, col_consensus_list


# ==== Correction ====
def corrected(faulty_model, critical_columns, refs_c, refs_nc, Indices, row_consensus_list, col_consensus_list, cos='reg'):
    faulty = faulty_model.model.clone().cpu().numpy()
    total_columns = faulty.shape[1]
    critical_columns = np.array(critical_columns)
    non_critical_columns = np.setdiff1d(np.arange(total_columns), critical_columns)

    C_c_err = faulty[:, critical_columns]
    C_nc_err = faulty[:, non_critical_columns]
    if cos != 'reg':
        err_det_nc = detect_errors(C_nc_err, refs_nc, cos_sim_custom)
    else:
        err_det_nc = detect_errors(C_nc_err, refs_nc, cos_sim)        
    C_nc_corrected = C_nc_err.copy()
    for i in err_det_nc:
        C_nc_corrected[:, i] = 0

    if cos != 'reg':
        err_det_c = detect_errors(C_c_err, refs_c, cos_sim_custom)
    else:
        err_det_c = detect_errors(C_c_err, refs_c, cos_sim)    
            
    C_c_corrected = C_c_err.copy()

    for i in err_det_c:
        for cluster, row_consensus, col_consensus in zip(Indices, row_consensus_list, col_consensus_list):
            if i in cluster:
                cluster_idx = cluster.tolist().index(i)
                for row in range(C_c_err.shape[0]):
                    rc = row_consensus[cluster_idx]
                    cc = col_consensus[row]
                    if not np.isnan(rc) and not np.isnan(cc):
                        if rc == cc:
                            C_c_corrected[row, i] = rc
                    elif not np.isnan(rc):
                        C_c_corrected[row, i] = rc
                    elif not np.isnan(cc):
                        C_c_corrected[row, i] = cc
                break

    corrected_matrix = np.zeros_like(faulty)
    corrected_matrix[:, critical_columns] = C_c_corrected
    corrected_matrix[:, non_critical_columns] = C_nc_corrected

    corrected_tensor = torch.tensor(corrected_matrix, dtype=torch.float32).to(faulty_model.model.device)
    corrected_model = copy.deepcopy(faulty_model)
    corrected_model.model = corrected_tensor
    return corrected_model


# ==== Load Data ====
def load_data(folder):
    print(f"+++ Loading {folder} +++")
    x = np.load(f'./data/{folder}/X_train.npy')
    y = np.load(f'./data/{folder}/y_train.npy')
    x_test = np.load(f'./data/{folder}/X_test.npy')
    y_test = np.load(f'./data/{folder}/y_test.npy')

    x, x_test = x.reshape(len(x), -1), x_test.reshape(len(x_test), -1)
    scaler = preprocessing.Normalizer().fit(x)
    x, x_test = scaler.transform(x), scaler.transform(x_test)

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y.flatten(), dtype=torch.long),
        torch.tensor(y_test.flatten(), dtype=torch.long),
        folder
    )

def compute_parity(col):
    return col.sum() % 2

def get_hamming_params(k):
    r = 0
    while (2 ** r) < (k + r + 1):
        r += 1
    return r

def insert_parity_bits(data, r):
    k = len(data)
    n = k + r
    result = torch.zeros(n, dtype=torch.int32)
    j = 0
    for i in range(n):
        if math.log2(i + 1).is_integer():
            continue
        result[i] = data[j]
        j += 1
    return result

def compute_parity_bits(codeword, r):
    n = len(codeword)
    for i in range(r):
        parity_pos = 2 ** i
        parity = 0
        for j in range(1, n + 1):
            if j & parity_pos and j != parity_pos:
                parity ^= codeword[j - 1]
        codeword[parity_pos - 1] = parity
    return codeword

def compute_overall_parity(codeword):
    return codeword.sum() % 2

def ecc_encode_generic(data):
    d = data.clone().int().squeeze()
    k = len(d)
    r = get_hamming_params(k)
    codeword = insert_parity_bits(d, r)
    codeword = compute_parity_bits(codeword, r)
    overall = compute_overall_parity(codeword)
    return torch.cat([codeword, overall.unsqueeze(0)]).unsqueeze(0)

def ecc_decode_generic(codeword, k):
    codeword = codeword.clone().int().squeeze()
    n = len(codeword) - 1
    r = get_hamming_params(k)
    syndrome = 0
    for i in range(r):
        parity_pos = 2 ** i
        parity = 0
        for j in range(1, n + 1):
            if j & parity_pos:
                parity ^= codeword[j - 1]
        if parity:
            syndrome += parity_pos
    if syndrome != 0 and syndrome <= n:
        codeword[syndrome - 1] ^= 1
    data = []
    for i in range(n):
        if not math.log2(i + 1).is_integer():
            data.append(codeword[i])
    return torch.tensor(data, dtype=torch.int32).unsqueeze(0)

def encode_ecc(model, sr_idx, lsr_idx):
    model_bin = ((1 + model.model) / 2).int()
    ECC_SR, Parity_LSR = [], []
    k = model_bin.shape[0]
    for idx in sr_idx:
        col = model_bin[:, idx].unsqueeze(0)
        ECC_SR.append(ecc_encode_generic(col))
    for idx in lsr_idx:
        parity = compute_parity(model_bin[:, idx])
        Parity_LSR.append(parity)
    return {
        'ECC_Codewords': torch.cat(ECC_SR, dim=0),
        'Parity_LSR': torch.tensor(Parity_LSR),
        'sr_idx': sr_idx,
        'lsr_idx': lsr_idx
    }

def decode_ecc(model, codes, corrupted_codewords=None):
    model_bin = ((1 + model.model) / 2).int().clone()
    if corrupted_codewords is None:
        corrupted_codewords = codes['ECC_Codewords'].clone()
    k = model_bin.shape[0]
    for i, idx in enumerate(codes['sr_idx']):
        corrected_data = ecc_decode_generic(corrupted_codewords[i].unsqueeze(0), k)
        model_bin[:, idx] = corrected_data.squeeze()
    for i, idx in enumerate(codes['lsr_idx']):
        parity = compute_parity(model_bin[:, idx])
        if parity != codes['Parity_LSR'][i]:
            model_bin[:, idx] = 0
    model.model = (2 * model_bin - 1).float()
    return model

def classify_columns_by_sensitivity(model, x_val, y_val, acc_threshold=0.95):
    base_model = copy.deepcopy(model)
    C, D = model.model.shape
    acc = (base_model(x_val) == y_val).float().mean().item()
    nom = acc_threshold * acc
    abs_sums = torch.abs(model.model.sum(dim=0))
    unique_sums = sorted(set(abs_sums.tolist()), reverse=True)
    mask = torch.ones(D, dtype=torch.bool)
    pruned_model = model.model.clone()
    for val in unique_sums[::-1]:
        cols_to_zero = torch.where(abs_sums == val)[0]
        temp_model = pruned_model.clone()
        temp_model[:, cols_to_zero] = 0
        base_model.model = temp_model
        acc = (base_model(x_val) == y_val).float().mean().item()
        if acc < nom:
            break
        pruned_model[:, cols_to_zero] = 0
        mask[cols_to_zero] = False
    sr_idx = torch.where(mask)[0].tolist()
    lsr_idx = torch.where(~mask)[0].tolist()
    return sr_idx, lsr_idx
# ==== Main ====
def main(INDX):
    #folders = ['ucihar','gtsrb','isolet','fashion_mnist','mnist']
    folders = [['ucihar','gtsrb','isolet','fashion_mnist','mnist'][INDX]]
    for folder in folders:
        x, x_test, y, y_test, name = load_data(folder)
        if folder =='ucihar':
            y_test -= 1
            y -= 1
        model = torch.load(f'./model_{folder}/model.pt')
        float_model = torch.load(f'./model_{folder}/pruned_model.pt')
        critical_columns = torch.load(f'./model_{folder}/kept_columns.pt')
        H_test = model.encode(x_test)
        
        model.model = torch.sign(float_model)
        model.model[model.model == 0] = -1
        base_model = copy.deepcopy(model)

        clean = base_model.model.clone().cpu().numpy()
        total_columns = clean.shape[1]
        critical_columns = np.array(critical_columns)
        non_critical_columns = np.setdiff1d(np.arange(total_columns), critical_columns)

        C_c = clean[:, critical_columns]
        C_nc = clean[:, non_critical_columns]

        val_ratio = 0.2
        dataset = TensorDataset(x, y)
        val_size = int(val_ratio * len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        x_val = torch.stack([val_set[i][0] for i in range(len(val_set))])
        y_val = torch.tensor([val_set[i][1] for i in range(len(val_set))])
        
        
        sr_idx, lsr_idx = classify_columns_by_sensitivity(model, x_val, y_val)
        code = encode_ecc(model, sr_idx, lsr_idx)
        decoded_model = decode_ecc(model, code)
        
        NN = 5
        results = {'faulty': [], 'ETS': [], 'OUR': [], 'OUR-Ablation': []}
        ER = list(range(6, 0, -1))+[0.3]
        for p in tqdm(ER, desc="Error rates"):
            er = 10 ** (-p)
            
            refs_c_clean = compute_cosine_refs(C_c, cos_sim_custom)
            refs_nc_clean = compute_cosine_refs(C_nc, cos_sim_custom)
            Indices, row_consensus_list_clean, col_consensus_list_clean = cluster_critical_with_cosine(C_c)

            refs_c = protect_vector_float(refs_c_clean, er)
            refs_nc = protect_vector_float(refs_nc_clean, er)
            row_consensus_list = [protect_vector(rc, er, N=5) for rc in row_consensus_list_clean]
            col_consensus_list = [protect_vector(cc, er, N=5) for cc in col_consensus_list_clean]
            
            
            
            refs_c_clean = compute_cosine_refs(C_c, cos_sim)
            refs_nc_clean = compute_cosine_refs(C_nc, cos_sim)
            Indices, row_consensus_list_clean, col_consensus_list_clean = cluster_critical_with_cosine(C_c)

            refs_c_r = protect_vector_float(refs_c_clean, er)
            refs_nc_r = protect_vector_float(refs_nc_clean, er)
            row_consensus_list_r = [protect_vector(rc, er, N=5) for rc in row_consensus_list_clean]
            col_consensus_list_r = [protect_vector(cc, er, N=5) for cc in col_consensus_list_clean]            

            scores = {k: 0 for k in results}
            for _ in range(NN):
                faulty_model = copy.deepcopy(base_model)
                faulty_model.model, _ = inject_error(faulty_model.model, er)

                with torch.no_grad():
                    scores['faulty'] += (faulty_model(H_test, encoded='True') == y_test).float().mean()

                corrected_model = corrected(
                    faulty_model, critical_columns, refs_c, refs_nc,
                    Indices, row_consensus_list, col_consensus_list
                    , cos='custom'
                )

                with torch.no_grad():
                    scores['OUR'] += (corrected_model(H_test, encoded='True') == y_test).float().mean()

                corrected_model = corrected(
                    faulty_model, critical_columns, refs_c_r, refs_nc_r,
                    Indices, row_consensus_list_r, col_consensus_list_r
                    , cos='reg'
                )


                with torch.no_grad():
                    scores['OUR-Ablation'] += (corrected_model(H_test, encoded='True') == y_test).float().mean()

                cw_clean = code['ECC_Codewords']
                cw_tmr = []
                for i in range(cw_clean.shape[0]):
                    c1, _ = inject_error(cw_clean[i], er)
                    c2, _ = inject_error(cw_clean[i], er)
                    c3, _ = inject_error(cw_clean[i], er)
                    c4, _ = inject_error(cw_clean[i], er)
                    c5, _ = inject_error(cw_clean[i], er)
                    voted = tmr_vote(c1, c2, c3, c4, c5)
                    cw_tmr.append(voted)
                cw_voted = torch.stack(cw_tmr)

                decoded_model = decode_ecc(copy.deepcopy(faulty_model), code, corrupted_codewords=cw_voted)
                scores['ETS'] += (decoded_model(H_test, encoded='True') == y_test).float().mean()
                

            for k in results:
                results[k].append(scores[k].item() / NN)

            print(f"\n[1e-{p}]")
            for k in results:
                print(f"{k.capitalize():<10}: {results[k][-1]:.4f}")
        
        for k, v in results.items():
            np.save(f"faulty_{k}_{name}.npy", v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main pipeline for a specific dataset.")
    parser.add_argument("index", type=int, choices=range(0, 5), help="Index of the dataset to use (0 to 4)")
    args = parser.parse_args()
    main(args.index)

