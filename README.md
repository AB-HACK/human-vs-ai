# AI vs Human Essay Classification (Local Terminal)

Classify essays as AI-generated or human-written using a TF‑IDF + Logistic Regression pipeline. This README assumes you will run everything from your system terminal (Windows PowerShell/CMD or macOS/Linux shell).

## 1) Requirements

- Python 3.10+ recommended
- Pip

Install Python packages:
```bash
pip install -r requirements.txt
```

If you use a virtual environment (optional):
- Windows PowerShell
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
- macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset

Use the Kaggle dataset: `balanced_ai_human_prompts.csv` from `human-vs-ai-generated-essays`.

Place the CSV anywhere on your machine. The scripts let you pass a path, or you can put it in the project root for convenience.

Expected columns:
- `text` — essay content
- `generated` — label (0 = Human, 1 = AI)

## 3) Train from Terminal

Two options:

- A) Run the ready-to-go analysis script (saves `model.pkl` and `vectorizer.pkl`):
```bash
python kaggle_analysis.py
```

- B) Run the modular training pipeline and pass your CSV path (recommended for flexibility):
```bash
python -c "from src.modeling.train import train_model, save_model; \
model, vect, Xt, yt, yp = train_model(file_path='balanced_ai_human_prompts.csv'); \
save_model(model, vect)"
```

Notes:
- Replace `'balanced_ai_human_prompts.csv'` with the actual path if the CSV is elsewhere.
- The training script will download NLTK data the first time it runs.

## 4) Predict from Terminal

After training, use either script below. Both expect `model.pkl` and `vectorizer.pkl` in the project root.

- Simple, interactive prediction:
```bash
python simple_predict.py
```

- Programmatic single prediction:
```bash
python -c "from src.modeling.predict import predict_single_text; \
print(predict_single_text('This is a sample essay about AI.'))"
```

## 5) Project Structure

```
human-vs-ai/
├── kaggle_analysis.py          # End‑to‑end training + visuals (local CSV supported)
├── run_analysis.py             # Original reference script
├── simple_predict.py           # Interactive/simple prediction
├── requirements.txt            # Dependencies
├── src/
│   ├── config.py              # Settings + NLTK setup
│   ├── dataset.py             # Data loading/splitting helpers
│   ├── features.py            # Text preprocessing + TF‑IDF
│   ├── plots.py               # Plots (class dist., confusion matrix, wordcloud)
│   └── modeling/
│       ├── train.py           # Training pipeline + save
│       └── predict.py         # Predict utils (single/batch/interactive)
└── README.md
```

## 6) Troubleshooting

- No module named nltk:
```bash
pip install nltk
```
Then run once to download resources (done automatically by scripts). If blocked on Windows PowerShell execution policy, use CMD (`cmd.exe`) or set policy for your user:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

- Different CSV column names:
Update your CSV to have `text` and `generated`, or adjust the loader in `src/dataset.py`.

## 7) Notes

- Randomness is controlled via `RANDOM_STATE` in `src/config.py` for reproducible splits and model behavior.
- The TF‑IDF configuration (bigrams, min/max df, max features) is set in `src/features.py`.

All commands above are intended to be run from your system terminal inside the project directory.
