import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = r"D:\Datasets\Iris.csv"

def load_dataset(path=DATA_PATH):
	if os.path.exists(path):
		try:
			df = pd.read_csv(path)
			return df
		except Exception as e:
			raise RuntimeError(f"Failed to read CSV at {path}: {e}")
	from sklearn.datasets import load_iris
	iris = load_iris()
	df = pd.DataFrame(iris.data, columns=iris.feature_names)
	df['Species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
	return df

def main():
	df = load_dataset()
	print(df.head())

	drop_cols = [c for c in ['Id', 'ID'] if c in df.columns]
	if 'Species' not in df.columns:
		raise RuntimeError("Dataset does not contain a 'Species' column and is not recognizable")

	X = df.drop(drop_cols + ['Species'], axis=1)
	y = df['Species']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("\nClassification Report:\n", classification_report(y_test, y_pred))

	try:
		sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
	except Exception:
		sample = X.iloc[[0]]

	prediction = model.predict(sample)
	print("\nPredicted species for sample:", prediction[0])

if __name__ == '__main__':
	main()