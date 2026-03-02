# Formative 3 — Probability Distributions, Bayesian Probability, and Gradient Descent

| Member | Part |
|--------|------|
| James Mukunzi | Part 1: Bivariate Normal Distribution |
| Favor | Part 2: Bayesian Probability |
| Chinemerem *(Lead)* | Part 3: Gradient Descent — Manual Calculation |
| Ryan | Part 4: Gradient Descent — Code Implementation |

---

## Project Structure

```
Formative3/
├── README.md                                   ← You are here
├── CONTRIBUTIONS.md                            ← Team responsibilities (source)
├── CONTRIBUTIONS.pdf                           ←  REQUIRED for submission
├── Formative3_Notebook.ipynb                   ←  PRIMARY DELIVERABLE
├── data/
│   ├── education_africa.csv                    ← Part 1 dataset (download below)
│   └── imdb_reviews.csv                        ← Part 2 dataset (download below)
├── formative3_utils/
│   ├── __init__.py
│   ├── data_loading.py                         ← Shared: loads both datasets
│   ├── distribution.py                         ← Part 1: Bivariate Normal (James)
│   ├── bayesian.py                             ← Part 2: Bayesian Probability (Favor)
│   ├── manual_calculator.py                    ← Part 3: GD verification (Chinemerem)
│   ├── gradient_descent.py                     ← Part 4: Coded GD (Ryan)
│   └── visualization.py                        ← Shared: visualisation helpers
├── manual_calculations/
│   └── Part3_Manual_Gradient_Descent.pdf       ←  REQUIRED: handwritten scans
└── requirements.txt
```

---

## ⬇ Dataset Downloads

### Part 1 — Education in Africa (James)

**Dataset:** Education in Africa  
**Download:** https://www.kaggle.com/datasets/lydia70/education-in-africa?select=Education+in+General.csv

Steps:
1. Click the link above and download `Education in General.csv`
2. Rename it to `education_africa.csv`
3. Place it inside the `data/` folder

> `data_loading.py` also auto-detects the original name `Education in General.csv`
> so renaming is optional — as long as the file is in `data/`.

---

### Part 2 — IMDb Movie Reviews (Favor)

**Dataset:** IMDb Dataset of 50K Movie Reviews  
**Download:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Steps:
1. Click the link above and download `IMDB Dataset.csv`
2. Rename it to `imdb_reviews.csv`
3. Place it inside the `data/` folder

> `data_loading.py` also auto-detects the original name `IMDB Dataset.csv`
> so renaming is optional — as long as the file is in `data/`.

---

##  Setup and Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download both datasets (see above) and place in data/

# 3. Open the notebook
jupyter notebook Formative3_Notebook.ipynb
```

Run all cells top-to-bottom. Every cell must show output.

---
## Team Task Sheet

https://github.com/Mukunzijames/Formative3/blob/main/Team%20Task%20Sheet_%5BFormative%203_C1_Group%205%20%5D%20-%201.pdf

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation — PDF formula, array operations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Contour plots, 3D surface, convergence charts |
| `seaborn` | Heatmap visualisation (Part 2) |
| `scipy` | Optimisation wrapper (Part 4) |
| `jupyter` | Notebook environment |
