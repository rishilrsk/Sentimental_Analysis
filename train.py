"""Train the SVM sentiment model and save it to sentiment_model.joblib."""
from sentiment import train

if __name__ == "__main__":
    train(save=True, verbose=True)
