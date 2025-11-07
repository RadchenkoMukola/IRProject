# News Classification with RoBERTa

## Project Overview
This project is focused on **classifying news articles** into categories such as **sports, economics, and politics** using a fine-tuned **RoBERTa** model.  
The team collaborates to clean data, visualize results, train the model, and create the final presentation.

---

## Dataset
- The dataset is provided in **JSON** and **CSV** formats.
- CSV columns:  
  `link, headline, category, short_description, authors, date`
- Raw dataset is stored in:  
  `Data/raw/`
- Cleaned and processed dataset is stored in:  
  `Data/processed/`

**Note:** The cleaned dataset removes unnecessary columns (`short_description`, `authors`, `date`) and includes a new column with the full extracted news text.

---

## Folder Structure

```
project-name/
├── .github/             # GitHub settings (CODEOWNERS)
├── data/
│   ├── raw/             # Original dataset
│   └── processed/       # Cleaned dataset ready for modeling
├── src/                 # Python scripts
│   ├── data_cleaning.py # Dataset cleaning functions
│   ├── visualization.py # Functions to generate plots/graphs
│   ├── train.py         # Training script for RoBERTa
│   ├── evaluate.py      # Model evaluation
│   └── predict.py       # Make predictions with trained model
├── results/
│   ├── figures/         # Plots and charts
│   ├── metrics/         # Model metrics
│   └── models/          # Saved models
├── presentation/
│   └── slides/          # Final presentation materials
├── requirements.txt     # Python dependencies
└── README.md
```
---

## Team Roles & Responsibilities

| Role | Responsibilities | Folder / Files |
|------|-----------------|----------------|
| **Code Cleaner** | - Clean dataset<br>- Remove unnecessary columns (`short_description`, `authors`, `date`)<br>- Extract news text (e.g., using `newspaper3k`) | `data/raw/` → `data/processed/`, `src/data_cleaning.py` |
| **Visualizer** | - Graph initial dataset (class distributions, text length, etc.)<br>- Graph model results (confusion matrix, accuracy, etc.) | `src/visualization.py`, outputs to `results/figures/` |
| **Presentation Maker** | - Use visuals from Visualizer to create presentation | `presentation/slides/` |
| **Leader / Model Trainer** | - Fine-tune RoBERTa<br>- Train, evaluate, and predict<br>- Save models and metrics | `src/train.py`, `src/predict.py`, `src/evaluate.py`, `results/models/`, `results/metrics/` |

---

## Workflow

1. **Code Cleaner** cleans the dataset → saves in `data/processed/`.  
2. **Visualizer** generates initial dataset visualizations → saves in `results/figures/`.  
3. **Leader** trains and evaluates RoBERTa → saves models, predictions, and metrics.  
4. **Visualizer** generates results visualizations (confusion matrices, metrics charts).  
5. **Presentation Maker** uses `results/figures/` to build final slides.

---

## How project will work

1. Install dependencies:
```bash
pip install -r requirements.txt

2.Run dataset cleaning:
python src/data_cleaning.py

3.Generate visuals:
python src/visualization.py(can do from the start to raw dataset + after it`s clean)
python src/evaluate.py(comparing after model made predictions)

4.Train model & Predictions:
python src/train.py
python src/predict.py

5.Presentation folder fully for presentaion maker


