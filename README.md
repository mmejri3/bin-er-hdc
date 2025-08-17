# bin-er-hdc  
**Error Resilience in Hyperdimensional Computing via Dimension Criticality and Cross-Clustering**  

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

