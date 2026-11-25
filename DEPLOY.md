# Sentiment Analyzer – web app

A Streamlit app where a user types a sentence and gets back **positive**, **negative** or **neutral**, a confidence score for each, and the words that decided it. It uses the same TF-IDF + calibrated Linear SVM as `SVM.ipynb`.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | The web page (Streamlit). |
| `sentiment.py` | Text cleaning, training, loading and prediction. |
| `train.py` | Retrains the model and saves `sentiment_model.joblib`. |
| `sentiment_model.joblib` | The trained model (about 190 KB). |
| `DailyDialog.csv` | Training data. The app retrains from it if the saved model is missing or can't be loaded. |
| `requirements.txt` | Python dependencies. |
| `.streamlit/config.toml` | Light theme and colours. Keep it next to `app.py` when you push. |

## Run it locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

To retrain after changing the data or settings, run `python train.py`.

## Deploy for free on Streamlit Community Cloud

1. Push these files to the root of your GitHub repo (e.g. `rishilrsk/Sentimental_Analysis`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, and main file `app.py`.
4. Click **Deploy**. After a minute or two you get a public URL like `https://<name>.streamlit.app`.

## Alternative: Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space and pick the **Streamlit** SDK.
2. Upload the same files (or push them with git).
3. The Space builds and serves the app automatically.

## About the model

- Labels: joy → positive; sadness, anger, fear → negative; neutral → neutral.
- About 60% of the training data is negative, which pushes everyday sentences toward "negative". `sentiment.py` partly corrects for this (`PRIOR_STRENGTH = 0.5`). Test accuracy is 79.7% (80.7% without the correction), and recall improves from 61% to 69% for positive and from 71% to 77% for neutral. Set `PRIOR_STRENGTH = 0` and run `python train.py` to get back the notebook's exact behaviour.
- It's a word-based model, so it can misread sarcasm, negation ("not bad") and sentences that mix positive and negative feelings.
