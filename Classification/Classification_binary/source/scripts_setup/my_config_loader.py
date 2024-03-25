# config_loader.py

import yaml
import datetime

def load_configuration_model_gen(file_path):
    try:
        with open(file_path) as f:
            config_file = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found")

    mlf_key = datetime.datetime.now().strftime("%y%m%d%H%M")

    # General parameters
    try:
        cv = config_file["General"]["cv"]
        random_state = config_file["General"]["random_state"]
        n_jobs = config_file["General"]["n_jobs"]
        n_iter = config_file["General"]["n_iter"]
        n_trials = config_file["General"]["n_trials"]
        init_points = config_file["General"]["init_points"]
        testing = config_file["General"]["testing"]
    except KeyError as e:
        raise KeyError(f"Missing key in General section: {e}")
        
    return mlf_key, \
           cv, random_state, n_jobs, n_iter, n_trials, init_points, testing

def load_configuration_mlflow(file_path):
    try:
        with open(file_path) as f:
            config_file = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found")

    mlf_key = datetime.datetime.now().strftime("%y%m%d%H%M")
    
    # MLFlow parameters
    try:
        mlf_tracking_server_uri = config_file["MLFlow"]["tracking_server_uri"]
        mlf_experiment_name = config_file["MLFlow"]["experiment_name"]
        mlf_project_name = config_file["MLFlow"]["project_name"]
        mlf_team = config_file["MLFlow"]["team"]
    except KeyError as e:
        raise KeyError(f"Missing key in MLFlow section: {e}")

    return mlf_tracking_server_uri, mlf_experiment_name, mlf_project_name, mlf_team

def load_configuration_model_svc(file_path):
    try:
        with open(file_path) as f:
            config_file = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found")

    # SVC range params
    try:
        c_min = config_file["SVC"]["c_min"]
        c_max = config_file["SVC"]["c_max"]
        gamma_min = config_file["SVC"]["gamma_min"]
        gamma_max = config_file["SVC"]["gamma_max"]
    except KeyError as e:
        raise KeyError(f"Missing key in SVC section: {e}")
    
    return c_min, c_max, gamma_min, gamma_max

def load_configuration_model_knn(file_path):
    try:
        with open(file_path) as f:
            config_file = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Configuration file not found")
    
    # KNN parameters
    try:
        n_neighbors_min = config_file["KNN"]["n_neighbors_min"]
        n_neighbors_max = config_file["KNN"]["n_neighbors_max"]
        leaf_size_min = config_file["KNN"]["leaf_size_min"]
        leaf_size_max = config_file["KNN"]["leaf_size_max"]
    except KeyError as e:
        raise KeyError(f"Missing key in KNN section: {e}")
    
    return n_neighbors_min, n_neighbors_max, leaf_size_min, leaf_size_max