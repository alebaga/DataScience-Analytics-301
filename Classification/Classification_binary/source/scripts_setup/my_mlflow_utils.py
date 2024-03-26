import numpy as np
import mlflow
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

# Logs Tags & parameters in MLFlow experiment
def mlf_log_tags_params_gen(param_tag, *args):
    try:
        # Ensure the number of arguments is even (tag_name, value)
        if len(args) % 2 != 0:
            raise ValueError("Arguments must be provided in pairs (param_name, value)")
        # Loop through pairs of arguments
        for i in range(0, len(args), 2):
            # Get tag name and value
            name = args[i]
            value = args[i + 1]
            if param_tag == "tag":          # Log tag
                mlflow.set_tag(name, value)
            elif param_tag == "param":      # Log parameter
                mlflow.log_param(name, value)
    except Exception as e:
        print(f"Error mlf_log_tags_params_generic: {e}")

# Logs metrics & model in MLFlow experiment
def mlf_log_metrics_models(class_report, model, tag, auc):
    try:
        mlflow.log_metric("accuracy", class_report["accuracy"])
        mlflow.log_metric("AUC", auc)
        
        for class_name, metrics in class_report.items():
            if class_name not in ["macro avg", "weighted avg"]:
                if isinstance(metrics, dict):  
                    for metric, value in metrics.items():
                        if metric in ["precision", "recall", "f1-score", "support"]:
                            mlflow.log_metric(f"{metric}_{class_name}", value)    
        #mlflow.log_figure(fig8, "qq_plot.png")
        mlflow.sklearn.log_model(model, tag) 
    except Exception as e:
        print(f"Error mlf_log_metrics_models: {e}")

# Logs AUC intervals in MLFlow experiment
# def log_metrics_auc_intervals(fpr, tpr):
#     try:
#         # Define the intervals (e.g., 10%, 20%, ..., 100%)
#         intervals = range(10, 101, 10)  # 10, 20, ..., 100        
#         # Initialize a dictionary to store the AUC for each interval
#         auc_intervals = {}        
#         # Compute the total number of data points
#         total_points = len(fpr)
        
#         # Loop through each interval
#         for interval in intervals:
#             # Calculate the index up to which the current interval falls
#             index = int((interval / 100) * total_points)            
#             # Extract the FPR and TPR values up to the current index
#             fpr_interval = fpr[:index + 1]
#             tpr_interval = tpr[:index + 1]            
#             # Compute the AUC for the current interval
#             auc_interval = auc(fpr_interval, tpr_interval)            
#             # Store the AUC for the current interval
#             auc_intervals[interval] = auc_interval
            
#         for interval, auc_interval in auc_intervals.items():
#             #print(f"AUC for {interval}%: {auc_interval}")  
#             mlflow.log_metric(f"AUC for {interval} perc", auc_interval)
#     except Exception as e:
#         print(f"Error log_metrics_auc_intervals: {e}")


def log_metrics_auc_intervals(fpr, tpr):
    try:
        # Combine fpr and tpr into a single 2D array
        points = np.column_stack((fpr, tpr))
        
        # Sort the points by the fpr in ascending order
        sorted_points = points[np.argsort(points[:, 0])]
        
        # Compute the total AUC
        total_auc = auc(sorted_points[:, 0], sorted_points[:, 1])
        
        # Initialize a dictionary to store the AUC for each interval
        auc_intervals = {}
        
        # Define the intervals (e.g., 10%, 20%, ..., 100%)
        intervals = range(10, 101, 10)
        
        # Compute AUC for each interval
        for interval in intervals:
            # Compute the index corresponding to the desired percentile
            index = int((interval / 100) * len(sorted_points))
            
            # Extract the points up to the current index
            interval_points = sorted_points[:index + 1]
            
            if len(interval_points) > 1:
                # Compute the AUC for the current interval
                auc_value = auc(interval_points[:, 0], interval_points[:, 1])
                
                # Store the AUC value for the current interval
                auc_intervals[interval] = auc_value
            else:
                auc_intervals[interval] = None

        for interval, auc_interval in auc_intervals.items():
            #print(f"AUC for {interval}%: {auc_interval}")  
            mlflow.log_metric(f"AUC for {interval} perc", auc_interval)
            #return auc_intervals, total_auc
        
    except Exception as e:
        print(f"Error in log_metrics_auc_intervals: {e}")
        return None, None

def mlf_log_tables(X_train, y_train, X_test, y_test):
    # Log an input datasets used for training & testing
    t_data = mlflow.data.from_numpy(X_train, targets=y_train) 
    mlflow.log_input(t_data, context="training")

    t_data = mlflow.data.from_numpy(X_test, targets=y_test) 
    mlflow.log_input(t_data, context="test")
