from sklearn.linear_model import LogisticRegression

from model_utils import train_evaluate_save_model


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

train_evaluate_save_model(model, "logistic_regression")