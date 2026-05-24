# bin-er-hdc  
**Efficient Multiple Error Correction in Binary HDC Systems Using Majority-Encoded Hypervector Matrices**  

[![Paper](https://img.shields.io/badge/paper-ATSITC%202025-blue)](./ATSITC-ASIA_2025_paper_110.pdf)  

---

## 📖 Overview  
This repository contains the official implementation of the **dimension-criticality and cross-clustering error resilience framework** for binary/bipolar Hyperdimensional Computing (HDC).  

Our approach:  
1. **Criticality Analysis (offline, via LNS)**  
   - Prunes non-critical dimensions of Class Hypervectors (CHVs) using **Local Neighborhood Search (LNS)**.  
   - Produces a pruned HDC model + list of critical columns.  

2. **Error Injection & Correction (online, via example.py)**  
   - Injects timing/soft errors (bit flips) into associative memory.  
   - Detects errors via **Encoding Check Hypervector (EChv)**.  
   - Applies **adaptive correction**:  
     - Non-critical dimensions → suppressed to zero.  
     - Critical dimensions → corrected via **row/column consensus from clustering**.  

Performance highlights:  
- **Up to 100× more resilient** than unprotected HDC.  
- **4× improvement** vs ECC under aggressive voltage scaling.  
- Only **7% energy/memory overhead**.  

---

## 🗂 Repository Structure  

```
bin-er-hdc/
│── example.py           # End-to-end error resilience simulation
│── LNS_removal.py       # Offline criticality analysis (LNS-based pruning)
│── baselines/           # Baseline error resilience (ECC, AbsSum, etc.)
│── datasets/            # Preprocessed data (MNIST, ISOLET, UCIHAR, etc.)
│── results/             # Output results (.npy, figures)
│── utils/               # Helper functions (encoding, clustering, metrics)
│── ATSITC-ASIA_2025_paper_110.pdf   # Paper with full methodology
│── README.md
```

---

## 🚀 Getting Started  

### 1. Installation  
```bash
git clone https://github.com/mmejri3/bin-er-hdc.git
cd bin-er-hdc
pip install -r requirements.txt
```

If no requirements file is present:  
```bash
pip install numpy torch matplotlib tqdm scikit-learn fxpmath onlinehd
```

---

### 2. Offline Stage: LNS-based Criticality Analysis  

Run **LNS_removal.py** to prune non-critical dimensions and generate:  
- `model_<dataset>/model.pt` → Original HDC model.  
- `model_<dataset>/pruned_model.pt` → LNS-pruned model.  
- `model_<dataset>/kept_columns.pt` → Critical columns indices.  

```bash
python LNS_removal.py
```

This will iterate over all datasets (`ucihar`, `isolet`, `gtsrb`, `fashion_mnist`, `mnist`).  

---

### 3. Online Stage: Error Resilience Simulation  

Run **example.py** to simulate error injection + correction.  

#### Usage  
```bash
python example.py <index>
```

where `<index>` selects dataset:  
- `0` → UCIHAR  
- `1` → GTSRB  
- `2` → ISOLET  
- `3` → Fashion-MNIST  
- `4` → MNIST  

Example:  
```bash
python example.py 4
```

#### What it does:  
- Loads pre-trained and LNS-pruned models.  
- Injects **bit-flip errors** into associative memory.  
- Applies different protection strategies:  
  - **Faulty (no protection)**  
  - **ECC baseline (ETS)**  
  - **Our method (custom cosine)**  
  - **Our ablation (regular cosine)**  
- Saves results as `.npy` arrays in `results/`.  

---

## 📊 Results  

- `LNS_removal.py` → Provides pruned critical dimensions, improving accuracy vs. Random/AbsSum pruning.  
- `example.py` → Simulates robustness under fault injection, demonstrating superiority of our proposed method.  

---

## 📜 Citation  

If you use this work, please cite our paper:  

```
@inproceedings{bin-er-hdc-2026,
  title={Efficient Multiple Error Correction in Binary HDC Systems Using Majority-Encoded Hypervector Matrices},
  booktitle={The 32nd IEEE International Symposium on On-Line Testing and Robust System Design},
  year={2026},
  author={Mohamed Mejri, Abhijit Chatterjee}
}
```
