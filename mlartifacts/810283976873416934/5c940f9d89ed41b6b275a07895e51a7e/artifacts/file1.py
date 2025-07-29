import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 12
n_estimators = 5

# Mention your exp below
mlflow.set_experiment('MLOps-Exp1') # If not exist, will create new exp
# or mention expID,
# with mlflow.start_run(experiment_id=197773580844288034):

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # creating a confusion_matrix plot 
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Metrix')

    # save plot 
    plt.savefig("Consfusion-matrix.png")

    # log artifacts using mlflow 
    mlflow.log_artifact("Consfusion-matrix.png")
    mlflow.log_artifact(__file__)

    # Tags
    mlflow.set_tags({"Author": "Sandeep", "Project": "wine classification"})

    # Infer the model signature
    signature = infer_signature(X_train, rf.predict(X_train))

    # Log the model
    mlflow.sklearn.log_model(
    rf,
    name="Random Forest Model",
    input_example=X_train[:5],
    signature=signature
    )

    # Output accuracy
    print(f"Accuracy: {accuracy:.4f}")

