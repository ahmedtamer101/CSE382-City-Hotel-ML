from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_dir = Path("data/processed/regression")
train = pd.read_csv(data_dir / "train_transformed.csv.gz")
val = pd.read_csv(data_dir / "val_transformed.csv.gz")

target = "ADR"
X_train = train.drop(columns=[target])
y_train = train[target]
X_val = val.drop(columns=[target]).reindex(columns=X_train.columns)
y_val = val[target]

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_val)

print("MAE:", mean_absolute_error(y_val, predictions))
print("RMSE:", mean_squared_error(y_val, predictions) ** 0.5)
print("R²:", r2_score(y_val, predictions))
