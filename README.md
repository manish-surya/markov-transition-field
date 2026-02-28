# 📡 Markov Transition Field for Time Series Analysis

Convert electricity load time series into 2D images using **Markov Transition Fields (MTF)**, with **Bayesian-optimized** parameters and full exploratory analysis.

---

## What This Does

Raw sensor readings are hard to analyse at scale and incompatible with image-based AI models. This project encodes a 1D time series into a 2D image that captures **both temporal ordering and probabilistic transition dynamics** — making it directly usable with CNNs or Vision Transformers.

```
Raw Signal → Discretize → Markov Transition Matrix → MTF Image → Downsample → Regime Analysis
```

---

## Dataset

**ETTh1** — Electricity Transformer Temperature  
- 17,420 hourly readings from a Chinese power grid  
- 7 features (load measurements + oil temperature)  
- Focus: `HUFL` (High Useful Load)  
- Source: [github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

---

## Key Features

- **Deep EDA** — distribution, ACF/PACF, correlation, outlier detection, temporal patterns
- **Bayesian Optimization** — Gaussian Process surrogate + Expected Improvement to find the optimal number of bins (maximises MTF Shannon entropy)
- **MTF Construction** — full 1000×1000 field + 100×100 block-averaged version
- **Regime Analysis** — MTF diagonal maps stable vs. volatile load periods onto the original signal

---

## Project Structure

```
├── markov_transition_field_enhanced_v2.ipynb   # Main notebook (EDA + BO + MTF)
├── MTF_Presentation.pptx                       # 28-slide visual presentation
├── MTF_Presentation.pdf                        # PDF version of presentation
└── README.md
```

---

## Quickstart

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-image pyts
```

Open `markov_transition_field_enhanced_v2.ipynb` and run all cells. The notebook will:
1. Load ETTh1 from GitHub
2. Run full EDA
3. Use Bayesian Optimization to find the optimal `n_bins`
4. Build and visualise the MTF
5. Produce the regime stability map

---

## Results

| Step | Output |
|------|--------|
| Optimal `n_bins` | Found via BO (GP + EI), maximising MTF entropy |
| ACF peaks | Lag 24 & 48 confirm daily 24-hour periodicity |
| MTF image | 1000×1000 → reduced to 100×100 via block averaging |
| Regime map | Stable (bright) vs. volatile (dark) segments on original signal |

---

## Applications

MTF images can be used as input to image classification models for:
- Fault & anomaly detection in power grids
- ECG / EEG classification in healthcare
- Market regime detection in finance
- Predictive maintenance in industrial IoT

---

## References

- Wang, Z. & Oates, T. (2015). *Encoding Time Series as Images for Visual Inspection and Classification.* AAAI.
- Brochu, E. et al. (2010). *A Tutorial on Bayesian Optimization.* arXiv:1012.2599
- [`pyts` library](https://pyts.readthedocs.io)
