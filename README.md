# TwoStreamNet

A dual-stream, gated, residual reformulation of the [ANI-1](https://pubs.rsc.org/en/content/articlelanding/2017/sc/c6sc05720a) atomic neural network potential, built for the BioEng 142 final project. TwoStreamNet splits the Atomic Environment Vector (AEV) into its radial and angular halves, processes each half with its own multi-layer encoder, fuses the two streams with a learned squeeze-and-excitation gate, and passes the gated representation through a residual stack before predicting per-atom energies.

Trained and evaluated on the ANI-1 dataset (organic molecules with H, C, N, O), TwoStreamNet achieves a held-out MAE of **4.236 ± 0.397 kcal/mol** under 3-fold cross-validation.

---

## Architecture

```
                ┌─ Radial stream  (Linear → LN → CELU) ×2 ─┐
AEV (N, 384) ──┤                                          ├─ Concat ─ SE gate ─ Residual stack ─ Output head ─ Σ E_atom
                └─ Angular stream (Linear → LN → CELU) ×2 ─┘
```

- **Two streams.** The 384-dim AEV is split at its known radial/angular boundary (64 radial, 320 angular) so each half is normalized and projected on its own natural scale.
- **Squeeze-and-Excitation gate.** A small two-layer MLP (reduction ratio `r=4`) produces a sigmoid-bounded mask that up-weights informative AEV channels and suppresses irrelevant ones.
- **Residual stack.** `n_residual` pre-activation residual blocks (LN → CELU → Linear → LN → CELU → Dropout → Linear) keep the gradient signal healthy in the deeper post-fusion network.
- **CELU activations** throughout, with **LayerNorm** before every activation and **Xavier-uniform** weight initialization.

See Section 3 of the notebook for the full implementation.

---

## Repository Structure

```
.
├── BioEng142_nb2.ipynb     # Main notebook: model, tuning, training, CV
├── figures.py              # Standalone script that builds the paper figures
├── figures/                # Output directory for generated PDF figures
├── final_report.pdf        # LaTeX-formatted writeup of the project
└── README.md               # This file
```

---

## Requirements

The project was developed in Google Colab on an A100 GPU. The notebook installs its own dependencies in the first cell, but for reference the core stack is:

- Python 3.12
- `torch` (CUDA build)
- `torchani==2.2.3` (pinned)
- `optuna`
- `scikit-learn`
- `numpy`, `matplotlib`, `tqdm`

Install locally with:

```bash
pip install torch torchani==2.2.3 optuna scikit-learn numpy matplotlib tqdm
```

---

## Dataset

Training data is drawn from the [ANI-1 release](https://figshare.com/collections/_/3846712), which provides DFT energies and conformations for organic molecules of up to eight heavy atoms. The release is partitioned into eight HDF5 shards, `ani_gdb_s01.h5` through `ani_gdb_s08.h5`, which should be placed in a directory the notebook can reach (the notebook expects them in `/content/drive/MyDrive/ANI-1_release` by default).

Two preprocessing steps are applied automatically:
1. Single-atom self-energies are subtracted using the shifter from the pretrained ANI-1x model.
2. The 384-dim AEV is computed using the AEV computer from ANI-1x.

> **Note.** The current results were trained on the `s01`-`s04` subset only (molecules with up to 4 heavy atoms) due to a file-extraction issue with the larger shards. See Section 10 of the notebook for details.

---

## Usage

### Running the notebook

Open `BioEng142_nb2.ipynb` in Google Colab (or a local Jupyter environment with GPU access) and run the cells in order. The notebook is organized into ten sections:

| Section | Contents |
|---:|---|
| 1 | Import dependencies |
| 2 | Training and evaluation loop (`ANITrainer`) |
| 3 | TwoStreamNet model (backbone, SE gate, residual blocks, ANI wrapper) |
| 4 | Justification of regularization strategies |
| 5 | Bayesian hyperparameter tuning with Optuna |
| 6 | Final tuned-model training and evaluation |
| 7 | Multiple training runs across seeds |
| 8 | K-fold cross-validation |
| 9 | Final production-mode performance summary |
| 10 | Discussion and comparison to ANI-1 |

### Regenerating the paper figures

```bash
python figures.py
```

Edit the `OUT` variable at the top of `figures.py` to point at your local output directory before running.

---

## Results

Production-mode estimates from 3-fold cross-validation:

| Metric | Value |
|---|---|
| MAE (total energy) | **4.236 ± 0.397 kcal/mol** |
| RMSE (total energy) | 6.003 ± 0.289 kcal/mol |
| RMSE (per atom) | 0.805 ± 0.039 kcal/mol/atom |

Single-split multi-run estimates (5 seeds, same train/validation split):

| Metric | Value |
|---|---|
| MAE (total energy) | 4.035 ± 0.238 kcal/mol |
| RMSE (total energy) | 5.711 ± 0.300 kcal/mol |
| RMSE (per atom) | 0.777 ± 0.027 kcal/mol/atom |

Tuned hyperparameters (Optuna TPE sampler):

| Hyperparameter | Value |
|---|---|
| `learning_rate` | 4.07 × 10⁻⁴ |
| `batch_size` | 256 |
| `l2_weight_decay` | 1.22 × 10⁻⁴ |
| `hidden_dim` | 256 |
| `n_residual` | 3 |
| `dropout` | 0.156 |

For context, ANI-1 reports a held-out RMSE of 1.8 kcal/mol on the GDB-10 extensibility set, trained on the full eight-shard dataset. TwoStreamNet sits roughly 4 to 5× above ANI-1, with most of the residual gap attributable to the smaller training subset, the shorter training schedule, and the use of standard MSE rather than ANI-1's exponential cost function.

---

## References

1. Smith, J. S., Isayev, O., & Roitberg, A. E. (2017). ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost. *Chemical Science*, 8, 3192-3203.
2. Behler, J., & Parrinello, M. (2007). Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. *Physical Review Letters*, 98, 146401.
3. Gao, X., Ramezanghorbani, F., Isayev, O., Smith, J. S., & Roitberg, A. E. (2020). TorchANI: A Free and Open Source PyTorch Based Deep Learning Implementation of the ANI Neural Network Potentials. *Journal of Chemical Information and Modeling*, 60(7), 3408-3415.
4. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *CVPR*, 7132-7141.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *ECCV*, 630-645.
6. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD*, 2623-2631.

---

## Acknowledgments

Built for BioEng 142 (Computational Methods in Bioengineering) at UC Berkeley. The training data, AEV computer, and base trainer scaffolding are taken from the [TorchANI](https://github.com/aiqm/torchani) library; TwoStreamNet replaces only the per-atom network on top of that pipeline.
