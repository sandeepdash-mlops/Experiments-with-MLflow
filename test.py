import mlflow

# mlflow expecting this should be http/https, but this is in file format
# (.venv) PS C:\Users\sandeep.d\Desktop\OPS-Series\Experiments-with-MLFlow> python .\src\test.py 
# file:///C:/Users/sandeep.d/Desktop/OPS-Series/Experiments-with-MLFlow/mlruns
# (.venv) PS C:\Users\sandeep.d\Desktop\OPS-Series\Experiments-with-MLFlow> 

# thats why the error
# file format tracking uri is a bug from mlflow end
#   File "C:\Users\sandeep.d\Desktop\OPS-Series\Experiments-with-MLFlow\.venv\Lib\site-packages\mlflow\store\artifact\mlflow_artifacts_repo.py", line 36, in _validate_uri_scheme
#     raise MlflowException(
#     ...<8 lines>...
#     )
# mlflow.exceptions.MlflowException: When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI, but it was currently set to file:///C:/Users/sandeep.d/Desktop/OPS-Series/Experiments-with-MLFlow/mlruns. Perhaps you forgot to set the tracking URI to the running MLflow server. To set the tracking URI, use either of the following methods:
# 1. Set the MLFLOW_TRACKING_URI environment variable to the desired tracking URI. `export MLFLOW_TRACKING_URI=http://localhost:5000`
# 2. Set the tracking URI programmatically by calling `mlflow.set_tracking_uri`. `mlflow.set_tracking_uri('http://localhost:5000')`


# To make this file format uri ti http/https

print("printing tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print("printing new tracking URI scheme below")
print(mlflow.get_tracking_uri())
print("\n")