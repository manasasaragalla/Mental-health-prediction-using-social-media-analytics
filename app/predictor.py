
from pathlib import Path

import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_DIR = PROJECT_ROOT / "models" / "baseline_stress"
ROBERTA_DIR = PROJECT_ROOT / "models" / "roberta_stress_new"


class StressPredictor:

    def __init__(self):
        print("Loading Logistic Regression...")
        self.lr_model = joblib.load(
            BASELINE_DIR / "tfidf_logistic_stress.joblib"
        )

        print("Loading Linear SVM...")
        self.svm_model = joblib.load(
            BASELINE_DIR / "tfidf_linear_svm_stress.joblib"
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = None
        self.roberta_model = None

        print("Loading RoBERTa tokenizer and trained model...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(ROBERTA_DIR),
                local_files_only=True
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                str(ROBERTA_DIR),
                local_files_only=True
            )

            model.to(self.device)
            model.eval()

            self.tokenizer = tokenizer
            self.roberta_model = model

            print("RoBERTa loaded successfully!")

        except (OSError, ValueError) as exc:
            print(f"RoBERTa unavailable: {exc}")
            print("Continuing with Logistic Regression and Linear SVM.")

        print("Baseline models loaded successfully!")

    def predict(self, text):
        if not text or not text.strip():
            raise ValueError("Please enter a social media post.")

        text = text.strip()

        lr_prediction = int(self.lr_model.predict([text])[0])
        svm_prediction = int(self.svm_model.predict([text])[0])

        labels = {
            0: "No Stress",
            1: "Stress"
        }

        result = {
            "text": text,
            "logistic_regression": {
                "prediction": labels[lr_prediction]
            },
            "linear_svm": {
                "prediction": labels[svm_prediction]
            },
            "roberta": {
                "prediction": "Unavailable",
                "confidence": None,
                "no_stress_probability": None,
                "stress_probability": None
            }
        }

        if self.tokenizer is not None and self.roberta_model is not None:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                probabilities = torch.softmax(
                    outputs.logits, dim=-1
                )[0]

            prediction = int(torch.argmax(probabilities).item())

            result["roberta"] = {
                "prediction": labels[prediction],
                "confidence": round(
                    float(probabilities[prediction].item()), 4
                ),
                "no_stress_probability": round(
                    float(probabilities[0].item()), 4
                ),
                "stress_probability": round(
                    float(probabilities[1].item()), 4
                )
            }

        return result


if __name__ == "__main__":
    predictor = StressPredictor()

    sample_text = (
        "I have so many assignments and deadlines. "
        "I feel overwhelmed and cannot concentrate."
    )

    result = predictor.predict(sample_text)

    print("\n========== PREDICTION RESULTS ==========")
    print("Input:", result["text"])
    print("Logistic Regression:",
          result["logistic_regression"]["prediction"])
    print("Linear SVM:", result["linear_svm"]["prediction"])
    print("RoBERTa:", result["roberta"]["prediction"])
    print("RoBERTa Confidence:", result["roberta"]["confidence"])
    print("Stress Probability:",
          result["roberta"]["stress_probability"])
