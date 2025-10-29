import numpy as np
np.float = float
import pandas as pd
import os
from flaml import AutoML
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import Parallel, delayed, dump
import sys
import shap
import libpysal
from esda import Moran

class Kernel:
    """
    Kernel function specifications for GALAX.
    
    Computes spatial weights based on distance and kernel function for 
    geographically weighted analysis.
    
    Parameters
    ----------
    coords_i : array-like, shape (2,)
        Coordinates [x, y] of the focal location.
    coords : array-like, shape (n_samples, 2)
        Coordinates of all locations in the dataset.
    bw : int or float
        Bandwidth value. If integer, represents the number of nearest neighbors.
    function : str, default='bisquare'
        Type of kernel function to use. Currently supports 'bisquare'.
        
    Attributes
    ----------
    kernel : ndarray
        Computed kernel weights for all locations relative to coords_i.
    """
    def __init__(self, coords_i, coords, bw, function='bisquare'):
        self.coords_i = coords_i
        self.coords = coords
        self.bw = bw
        self.function = function.lower()
        self.kernel = self._compute_kernel()

    def local_cdist(self):
        """
        Compute Euclidean distance between focal point and all other points.
        
        Returns
        -------
        ndarray
            Array of distances from coords_i to each point in coords.
        """
        # Euclidean distance for projected coordinates
        return np.sqrt(np.sum((self.coords_i - self.coords)**2, axis=1))

    def _compute_kernel(self):
        """
        Compute kernel weights using bisquare kernel function.
        
        The bisquare kernel applies weights based on distance, with closer
        points receiving higher weights. Points beyond the bandwidth receive
        zero weight.
        
        Returns
        -------
        ndarray
            Kernel weights for all locations.
        """
        dvec = self.local_cdist()
        bandwidth = np.partition(dvec, int(self.bw)-1)[int(self.bw)-1] * 1.0000001
        zs = dvec / bandwidth
        
        kernel = (1 - zs**2)**2
        kernel[dvec >= bandwidth] = 0
            
        return kernel

def check_class_sizes(weights, y_values, min_samples):
    """
    Check if each location has at least two classes and each present class 
    has at least min_samples samples within its bandwidth.
    
    This function is critical for classification tasks to ensure that local
    models have sufficient data from multiple classes for training.
    
    Parameters
    ----------
    weights : array-like or libpysal.weights object
        Weight matrix for current bandwidth.
    y_values : array-like, shape (n_samples,)
        Target class labels.
    min_samples : int
        Minimum required samples per class within each location's neighborhood.
        
    Returns
    -------
    bool
        True if class size requirements are met for all locations, False otherwise.
    """
    n_locations = len(y_values)
    all_valid = True
    problem_locations = []
    
    for i in range(n_locations):
        # Get neighboring indices where weight > 0
        if isinstance(weights, libpysal.weights.KNN):
            neighbor_indices = weights.neighbors[i]
        else:
            neighbor_indices = np.where(weights[i] > 0)[0]

        if len(neighbor_indices) == 0:
            all_valid = False
            problem_locations.append({
                'location': i,
                'problems': ['No neighbors found within bandwidth'],
                'class_counts': {}
            })
            continue
            
        neighbor_classes = y_values[neighbor_indices]
        
        # Get class counts for this location
        unique_local_classes, class_counts = np.unique(neighbor_classes, return_counts=True)
        class_count_dict = dict(zip(unique_local_classes, class_counts))

        location_valid = True
        problems = []
        
        # Check requirements:
        # 1. At least 2 classes present
        if len(unique_local_classes) < 2:
            location_valid = False
            problems.append(f"Only {len(unique_local_classes)} classes present")
            
        # 2. Each present class must have at least min_samples samples
        insufficient_classes = []
        for cls, count in class_count_dict.items():
            if count < min_samples:
                location_valid = False
                insufficient_classes.append((cls, count))
        
        if insufficient_classes:
            problems.append(f"Classes with insufficient samples: {insufficient_classes}")
        
        if not location_valid:
            all_valid = False
            problem_locations.append({
                'location': i,
                'problems': problems,
                'class_counts': class_count_dict
            })
            
    if not all_valid:
        print(f"\nFound {len(problem_locations)} locations with class size issues:")
        for loc in problem_locations[:5]:  # Show first 5 problem locations
            print(f"Location {loc['location']}:")
            print(f"  Problems: {', '.join(loc['problems'])}")
            print(f"  Class counts: {loc['class_counts']}")
        if len(problem_locations) > 5:
            print(f"... and {len(problem_locations) - 5} more locations with issues")
            
    return all_valid

def search_bw_lw_ISA(X, y, coords, bw_min=None, bw_max=None, step=1, kernel='bisquare', task='regression', min_samples_per_class=5):
    """
    Search for optimal bandwidth using Incremental Spatial Autocorrelation (ISA).
    
    This function identifies the optimal bandwidth by testing multiple bandwidths
    and selecting the one with the highest spatial autocorrelation (measured by
    Moran's I statistic). For classification tasks, it ensures each bandwidth
    provides sufficient samples per class.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,) or (n_samples, 1)
        Target values. For classification, these should be class labels.
    coords : array-like, shape (n_samples, 2)
        Spatial coordinates [x, y] for each sample.
    bw_min : int, optional
        Minimum bandwidth to test. If None, defaults to max(5% of samples, 
        n_features + 2, 20).
    bw_max : int, optional
        Maximum bandwidth to test. If None, defaults to max(95% of samples,
        n_features + 2).
    step : int, default=1
        Step size for bandwidth search.
    kernel : str, default='bisquare'
        Kernel function type (currently only 'bisquare' is supported).
    task : {'regression', 'classification'}, default='regression'
        Type of modeling task.
    min_samples_per_class : int, default=5
        For classification tasks, minimum required samples per class at each
        location.
        
    Returns
    -------
    found_bandwidth : int
        Optimal bandwidth value.
    found_moran_I : float
        Moran's I statistic at the optimal bandwidth.
    found_p_value : float
        P-value for the Moran's I statistic.
    """
    n_samples = X.shape[0]
    n_vars = X.shape[1]

    if bw_min is None:
        bw_min = max(round(n_samples * 0.05), n_vars + 2, 20)
    if bw_max is None:
        bw_max = max(round(n_samples * 0.95), n_vars + 2)

    print(f"\nStarting bandwidth search:")
    print(f"- Minimum bandwidth: {bw_min}")
    print(f"- Maximum bandwidth: {bw_max}")
    print(f"- Step size: {step}")
    if task == 'classification':
        print(f"- Minimum samples per class: {min_samples_per_class}")
    print("-" * 50)

    # Build k-d tree for efficient nearest neighbor search
    coords_array = np.array(coords)
    kd = libpysal.cg.KDTree(coords_array)

    # Initialize lists for storing results
    bandwidth_list = []
    moran_I_list = []
    z_score_list = []
    p_value_list = []

    # For classification tasks, use one-hot encoding
    if task == 'classification':
        print("Task: Classification")
        print(f"Number of samples: {len(y)}")
        print(f"Unique classes: {np.unique(y)}")
        y_reshaped = y.reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False)
        y_onehot = encoder.fit_transform(y_reshaped)
        n_classes = y_onehot.shape[1]
        print(f"Number of classes: {n_classes}")
        print("-" * 50)

    total_bandwidths = (bw_max - bw_min) // step + 1
    accepted_bandwidths = 0

    # Iterate through bandwidth values
    for current_bw in range(bw_min, bw_max + 1, step):
        # Create weights matrix
        print(f"\nTesting bandwidth {current_bw} ({(current_bw - bw_min) // step + 1}/{total_bandwidths})...")
        w = libpysal.weights.KNN(kd, current_bw)
        
        if task == 'classification':
            # Check if bandwidth provides enough samples per class
            has_enough_samples = check_class_sizes(w.full()[0], y, min_samples_per_class)
            
            if not has_enough_samples:
                print(f"✗ Bandwidth {current_bw} rejected: insufficient samples per class")
                continue

            accepted_bandwidths += 1
            print(f"✓ Bandwidth {current_bw} accepted: class size requirements met")
                
            # Calculate Moran's I for each class and average
            morans = []
            zscores = []
            pvalues = []
            
            for class_idx in range(n_classes):
                class_values = y_onehot[:, class_idx]
                moran = Moran(class_values, w)
                morans.append(moran.I)
                zscores.append(moran.z_norm)
                pvalues.append(moran.p_norm)
            
            # Average statistics across classes
            moran_I = np.mean(morans)
            z_score = np.mean(zscores)
            p_value = np.mean(pvalues)
        else:
            moran = Moran(y, w)
            moran_I = moran.I
            z_score = moran.z_norm
            p_value = moran.p_norm
            pass

        bandwidth_list.append(current_bw)
        moran_I_list.append(moran_I)
        z_score_list.append(z_score)
        p_value_list.append(p_value)
        
    print(f"\nBandwidth search completed:")
    print(f"- Tested {total_bandwidths} bandwidths")
    print(f"- {accepted_bandwidths} bandwidths accepted")
    print(f"- {total_bandwidths - accepted_bandwidths} bandwidths rejected")

    if not bandwidth_list:
        raise ValueError("No bandwidth found that satisfies the minimum class size requirements")

    # Find optimal bandwidth (highest z-score with p < 0.05)
    significant_indices = [i for i, p in enumerate(p_value_list) if p < 0.05]
    if significant_indices:
        # Use highest z-score with p < 0.05
        max_index = max(significant_indices, key=lambda i: z_score_list[i])
        found_bandwidth = bandwidth_list[max_index]
        found_moran_I = moran_I_list[max_index]
        found_p_value = p_value_list[max_index]
    else:
        print(f"\nNo significant bandwidth found (p < 0.05). Using lowest p-value bandwidth.")
        min_p_index = min(range(len(p_value_list)), key=lambda i: p_value_list[i])
        found_bandwidth = bandwidth_list[min_p_index]
        found_moran_I = moran_I_list[min_p_index]
        found_p_value = p_value_list[min_p_index]
   
    print(f"\nOptimal bandwidth found: {found_bandwidth}")
    print(f"Moran's I: {found_moran_I:.4f}")
    print(f"p-value: {found_p_value:.4f}")
    
    return found_bandwidth, found_moran_I, found_p_value

class GALAX:
    """
    GALAX: Geospatial Analysis Leveraging AutoML and eXplainable AI.
    
    Parameters
    ----------
    coords : array-like, shape (n_samples, 2)
        Spatial coordinates [x, y] for each sample location.
    y : array-like, shape (n_samples,) or (n_samples, 1)
        Target variable. For regression, continuous values; for classification, class labels.
    X : array-like, shape (n_samples, n_features)
        Feature matrix (should be standardized before input).
    bw : int, float, or {'isa'}, optional
        Bandwidth specification:
        - int or float: Fixed bandwidth (number of nearest neighbors)
        - 'isa': Automatic bandwidth selection using ISA.
    kernel : str, default='bisquare'
        Kernel function type. Currently supports 'bisquare'.
    automl_settings : dict, optional
        Settings for FLAML AutoML. If None, uses default settings appropriate for the task. 
        Common settings include:
        - 'time_budget': Time in seconds for AutoML search (default: 180)
        - 'estimator_list': List of algorithms to try (default: ['rf', 'xgboost', 'xgb_limitdepth', 'extra_tree'])
        - 'metric': Evaluation metric ('r2' for regression, 'accuracy' for classification)
        - 'seed': Random seed for reproducibility
    n_jobs : int, optional
        Number of parallel jobs for fitting local models. If None, uses 50% of available CPU cores.
    x_vars : list of str, optional
        Names of feature variables for interpretability and visualization.
    task : {'regression', 'classification'}, default='regression'
        Type of modeling task.
        
    Attributes
    ----------
    coords : ndarray
        Spatial coordinates of samples.
    y : ndarray
        Target values.
    X : ndarray
        Feature matrix.
    bw : int or float
        Final bandwidth value (after ISA selection if applicable).
    kernel : str
        Kernel function type.
    x_vars : list
        Feature variable names.
    task : str
        Modeling task type.
    automl_settings : dict
        AutoML configuration settings.
    n_jobs : int
        Number of parallel jobs.
    """
    def __init__(self, coords, y, X, bw=None, kernel='bisquare',  automl_settings=None, n_jobs=None, x_vars=None, task='regression'):
        self.coords = np.array(coords)
        self.y = np.array(y)
        self.X = np.array(X)
        self.bw = bw
        self.kernel = kernel
        self.x_vars = x_vars
        self.task = task

        # Validate bw parameter
        if isinstance(bw, str) and bw not in ['isa']:
            raise ValueError(f"Invalid bandwidth method: '{bw}'. Must be 'isa'.")

        # Set default AutoML settings based on task
        default_settings = {
            "time_budget": 180,
            "estimator_list": ['rf', 'xgboost', 'xgb_limitdepth', 'extra_tree'],
            "task": task,
            "metric": 'accuracy' if task == 'classification' else 'r2',
            "seed": 42,
            "verbose": 0,
        }
        self.automl_settings = {**default_settings, **(automl_settings or {})}
        total_cpus = os.cpu_count()
        default_jobs = int(total_cpus * 0.5)
        self.n_jobs = n_jobs if n_jobs is not None else default_jobs

    def _build_wi(self, i):
        """
        Build weight matrix for location i.
        
        Parameters
        ----------
        i : int
            Index of the focal location.
            
        Returns
        -------
        ndarray
            Kernel weights for location i relative to all other locations.
        """
        kernel_obj = Kernel(self.coords[i], self.coords, self.bw, function=self.kernel)
        return kernel_obj.kernel

    def fit(self):
        """
        Fit GALAX model by training local models at each location.
        
        Returns
        -------
        GALAXResults
            Object containing model results, predictions, SHAP values, and performance metrics.
        """
        if isinstance(self.bw, (int, float)):
            print(f"Using provided bandwidth: {self.bw}")
        elif self.bw == 'isa':
            try:
                print("Starting ISA bandwidth selection...")
                self.bw, moran_i, p_val = search_bw_lw_ISA(
                    X=self.X,
                    y=self.y, 
                    coords=self.coords,
                    kernel=self.kernel,
                    task=self.task,
                    min_samples_per_class=5
                )
                print(f"ISA bandwidth selection successful:")
                print(f"- Optimal bandwidth: {self.bw}")
                print(f"- Moran's I: {moran_i:.4f}")
                print(f"- p-value: {p_val:.4f}")
            except Exception as e:
                raise ValueError(f"ISA bandwidth search failed: {str(e)}")

        # Continue with the process_location and parallel processing
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_location)(i) 
            for i in range(len(self.y))
        )

        # Count successful locations
        successful_results = [r for r in results if r is not None]
        total_locations = self.X.shape[0]
        print(f"Total Successfully processed location: {len(successful_results)} / {total_locations}")
        
        if len(successful_results) == 0:
            raise ValueError("No locations were successfully processed. Check your data and AutoML settings.")
        
        return GALAXResults(self, successful_results)

    def _process_location(self, i):
        """
        Process a single location by fitting a local model.
        
        Parameters
        ----------
        i : int
            Index of the location to process.
            
        Returns
        -------
        dict or None
            Dictionary containing:
            - 'location_index': Location index
            - 'model': Trained model
            - 'estimator_name': Name of best estimator
            - 'local_metric': Local R² (regression) or accuracy (classification)
            - 'prediction': Prediction for location i
            - 'raw_shap_values_neighbors': SHAP values for neighbors
            - 'X_neighbors_values': Feature values of neighbors
            - 'y_neighbors_values': Target values of neighbors
            - 'weights_neighbors': Kernel weights of neighbors
            - Additional metrics specific to task type
            
            Returns None if processing fails.
        """
        try:
            # Get weights for location i
            weights_i = self._build_wi(i)
            neighbors_indices = np.where(weights_i > 0)[0]
            
            # Get data for local model
            X_neighbors = self.X[neighbors_indices]
            y_neighbors = self.y[neighbors_indices]
            weights_neighbors = weights_i[neighbors_indices]
            
            # Train local model for location i
            automl = AutoML()
            automl.fit(X_neighbors, y_neighbors.ravel(), sample_weight=weights_neighbors, **self.automl_settings)

            # Get predictions for all neighbors using location i's model
            y_pred_neighbors = automl.predict(X_neighbors)

            # Get SHAP values for the best model
            explainer = shap.TreeExplainer(automl.model.estimator)
            raw_shap_values = explainer.shap_values(X_neighbors)

            # Store the raw SHAP values directly.
            if isinstance(raw_shap_values, list):
                raw_shap_values_serializable = [s.tolist() for s in raw_shap_values]
            else: # If it's a numpy array (2D or 3D)
                raw_shap_values_serializable = raw_shap_values.tolist()
            
            # Store X_neighbors for original feature values.
            X_neighbors_serializable = X_neighbors.tolist()
            
            # Calculate metrics based on task type
            if self.task == 'classification':
                weighted_acc = np.sum(weights_neighbors * (y_neighbors.ravel() == y_pred_neighbors)) / np.sum(weights_neighbors)
                
                labels = np.unique(np.concatenate([y_neighbors, y_pred_neighbors]))
                labels = labels[~pd.isna(labels)]
                
                precision_per_class = precision_score(y_neighbors, y_pred_neighbors,
                                                      average=None, labels=labels, zero_division=np.nan)
                recall_per_class = recall_score(y_neighbors, y_pred_neighbors,
                                                average=None, labels=labels, zero_division=np.nan)
                f1_per_class = f1_score(y_neighbors, y_pred_neighbors,
                                        average=None, labels=labels, zero_division=np.nan)
                
                precision = np.nanmean(precision_per_class)
                recall = np.nanmean(recall_per_class)
                f1 = np.nanmean(f1_per_class)
                
                local_metric = weighted_acc
                additional_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'precision_per_class': precision_per_class.tolist(),
                    'recall_per_class': recall_per_class.tolist(),
                    'f1_per_class': f1_per_class.tolist(),
                    'classes_present': labels.tolist()
                }
            else: # Regression
                y_bar_i = np.sum(weights_neighbors * y_neighbors.ravel()) / np.sum(weights_neighbors)
                TSS_i = np.sum(weights_neighbors * (y_neighbors.ravel() - y_bar_i) ** 2)
                RSS_i = np.sum(weights_neighbors * (y_neighbors.ravel() - y_pred_neighbors) ** 2)
                local_r2_i = 1 - (RSS_i / TSS_i) if TSS_i != 0 else 0
                local_rmse_i = np.sqrt(
                    np.sum(weights_neighbors * (y_neighbors.ravel() - y_pred_neighbors) ** 2) /
                    np.sum(weights_neighbors)
                )
                local_metric = local_r2_i
                additional_metrics = {
                    'local_rmse': local_rmse_i
                }
            
            # Get prediction for location i using its own model
            pred_i = automl.predict(self.X[i].reshape(1, -1))[0]

            # Save the raw SHAP values and X_neighbors.
            location_results = {
                'location_index': i,
                'model': automl.model.estimator, 
                'estimator_name': automl.best_estimator,
                'local_metric': local_metric,
                'prediction': pred_i,
                'raw_shap_values_neighbors': raw_shap_values_serializable, # Stores raw SHAP output from explainer
                'X_neighbors_values': X_neighbors_serializable,           # Original feature values of neighbors
                'y_neighbors_values': y_neighbors.tolist(),
                'weights_neighbors': weights_neighbors.tolist(),
            }
            location_results.update(additional_metrics)
            print(f"Location {i}/{self.X.shape[0]} successfully trained ML model")
            
            return location_results
            
        except Exception as e:
            print(f"Error at location {i}: {str(e)}", file=sys.stderr)
            return None

class GALAXResults:
    """
    Results container for GALAX model outputs.
    
    Parameters
    ----------
    model : GALAX
        The fitted GALAX model instance.
    results : list of dict
        List of location-specific results from model fitting.
        
    Attributes
    ----------
    model : GALAX
        Reference to the fitted GALAX model.
    results : list of dict
        Raw results from each location.
    params : ndarray
        Predictions for each location.
    local_metrics : ndarray
        Local performance metrics (R² for regression, accuracy for classification).
    raw_shap_values_neighbors : list
        SHAP values for neighbors at each location.
    X_neighbors_values : list
        Feature values for neighbors at each location.
    y_neighbors_values : list
        Target values for neighbors at each location.
    weights_neighbors : list
        Kernel weights for neighbors at each location.
    location_original_indices : list
        Original indices of successfully processed locations.
        
    For regression tasks:
    local_rmse : ndarray
        Local RMSE at each location.
    global_r2 : float
        Global R² score across all locations.
    global_rmse : float
        Global RMSE across all locations.
        
    For classification tasks:
    local_precision : ndarray
        Local precision at each location.
    local_recall : ndarray
        Local recall at each location.
    local_f1 : ndarray
        Local F1 score at each location.
    global_accuracy : float
        Global accuracy across all locations.
    global_precision : float
        Global precision (weighted average) across all locations.
    global_recall : float
        Global recall (weighted average) across all locations.
    global_f1 : float
        Global F1 score (weighted average) across all locations.
    """
    def __init__(self, model, results):
        self.model = model
        self.results = results
        self._process_results()

        if self.model.task == 'regression':
            self.local_rmse = np.array([r['local_rmse'] for r in self.results])
        
    def _process_results(self):
        """
        Process and aggregate results from all locations.
        
        Extracts predictions, metrics, and SHAP values from raw results.
        Calculates global performance metrics based on task type.
        """
        self.params = np.array([r['prediction'] for r in self.results])
        self.local_metrics = np.array([r['local_metric'] for r in self.results])
        
        # Store raw SHAP values and X_neighbors for detailed analysis
        self.raw_shap_values_neighbors = []
        self.X_neighbors_values = []
        self.y_neighbors_values = []
        self.weights_neighbors = []
        self.location_original_indices = [r['location_index'] for r in self.results] # Store original indices

        for r in self.results:
            # Convert back to numpy arrays for consistency
            if isinstance(r['raw_shap_values_neighbors'], list) and all(isinstance(item, list) for item in r['raw_shap_values_neighbors']):
                self.raw_shap_values_neighbors.append([np.array(s) for s in r['raw_shap_values_neighbors']])
            else: 
                self.raw_shap_values_neighbors.append(np.array(r['raw_shap_values_neighbors']))
            
            self.X_neighbors_values.append(np.array(r['X_neighbors_values']))
            self.y_neighbors_values.append(np.array(r['y_neighbors_values']))
            self.weights_neighbors.append(np.array(r['weights_neighbors']))

        if self.model.task == 'classification':
            self.local_precision = np.array([r['precision'] for r in self.results])
            self.local_recall = np.array([r['recall'] for r in self.results])
            self.local_f1 = np.array([r['f1'] for r in self.results])
            
            # Calculate global classification metrics
            y_pred = self.params
            y_true = self.model.y[self.location_original_indices] # Use original indices
            
            if len(y_true) == 0 or len(y_pred) == 0:
                self.global_accuracy = np.nan
                self.global_precision = np.nan
                self.global_recall = np.nan
                self.global_f1 = np.nan
            else:
                self.global_accuracy = accuracy_score(y_true, y_pred)
                self.global_precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
                self.global_recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
                self.global_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
        else: # Regression
            y_pred = self.params.reshape(-1, 1)
            y_true = self.model.y[self.location_original_indices].reshape(-1, 1) # Use original indices
            
            if len(y_true) == 0 or len(y_pred) == 0:
                self.global_r2 = np.nan
                self.global_rmse = np.nan
            else:
                self.global_r2 = r2_score(y_true, y_pred)
                self.global_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    def summary(self):
        """
        Print a summary of model results.
        """
        print(f"GALAX Results Summary")
        print("-" * 50)
        print(f"Task: {self.model.task}")
        print(f"Bandwidth: {self.model.bw}")
        if self.model.task == 'classification':
            print(f"Global Accuracy: {self.global_accuracy:.4f}")
            print(f"Global Precision: {self.global_precision:.4f}")
            print(f"Global Recall: {self.global_recall:.4f}")
            print(f"Global F1 Score: {self.global_f1:.4f}")
            print(f"Local Precision Statistics:")
            print(f"  - Mean: {np.mean(self.local_precision):.4f}")
            print(f"  - Min: {np.min(self.local_precision):.4f}")
            print(f"  - Max: {np.max(self.local_precision):.4f}")
            print(f"  - Std: {np.std(self.local_precision):.4f}")
        else:
            print(f"Global R²: {self.global_r2:.4f}")
            print(f"Global RMSE: {self.global_rmse:.4f}")
            print(f"Local R² Statistics:")
            print(f"  - Mean: {np.mean(self.local_metrics):.4f}")
            print(f"  - Min: {np.min(self.local_metrics):.4f}")
            print(f"  - Max: {np.max(self.local_metrics):.4f}")
            print(f"  - Std: {np.std(self.local_metrics):.4f}")
            print(f"Local RMSE Statistics:")
            print(f"  - Mean: {np.mean(self.local_rmse):.4f}")
            print(f"  - Min: {np.min(self.local_rmse):.4f}")
            print(f"  - Max: {np.max(self.local_rmse):.4f}")
            print(f"  - Std: {np.std(self.local_rmse):.4f}")

    def save_results(self, filename):
        """
        Save results to a file using joblib.
        
        Parameters
        ----------
        filename : str
            Path where results will be saved. Should end with '.joblib'.
        """
        successful_results = [r for r in self.results if r is not None]

        total_locations = len(self.model.coords)
        successful_locations = len(successful_results)
        results_dict = {
            'task': self.model.task,
            'bandwidth': self.model.bw,
            'predictions': self.params.tolist(),
            'coords': self.model.coords.tolist(),
            'location_results': successful_results,
            'x_variables': self.model.x_vars if self.model.x_vars else [],
            'total_locations': total_locations,
            'successful_locations': successful_locations
        }

        if self.model.task == 'classification':
            results_dict.update({
                'global_accuracy': self.global_accuracy,
                'global_precision': self.global_precision,
                'global_recall': self.global_recall,
                'global_f1': self.global_f1,
                'local_precision': self.local_precision.tolist(),
                'local_recall': self.local_recall.tolist(),
                'local_f1': self.local_f1.tolist()
            })
        else:
            results_dict.update({
                'global_r2': self.global_r2,
                'global_rmse': self.global_rmse,
                'local_r2': self.local_metrics.tolist(),
                'local_rmse': self.local_rmse.tolist()
            })
        
        dump(results_dict, filename)

def main():
    ############## Regression ##############
    if False:
        ############# test data #############
        data = pd.read_csv("../data/311Request.csv")
        columns_to_exclude = ['CBG ID', 'Lon', 'Lat', '311_requests']
        x_vars = [column for column in data.columns if column not in columns_to_exclude]
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(data[x_vars])
        y = data['311_requests'].values.reshape(-1, 1)
        coords = data[['Lon', 'Lat']].values
        
        automl_settings = {
            "time_budget": 180,
            "estimator_list": ['rf', 'xgboost', 'xgb_limitdepth', 'extra_tree'],
            "task": 'regression',
            "metric": 'r2',
            "seed": 42,
            "verbose": 0,
        }
        
        model = GALAX(
            coords=coords,
            y=y,
            X=X,
            bw='isa',
            kernel='bisquare',
            automl_settings=automl_settings,
            n_jobs=48,
            x_vars=x_vars
        )

    ############## Classification ##############
    if True:
        ############# test data #############
        data = pd.read_csv("../data/311Request_class.csv")
        columns_to_exclude = ['CBG ID', 'Lon', 'Lat', '311_requests']
        x_vars = [column for column in data.columns if column not in columns_to_exclude]
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(data[x_vars])
        y = data['311_requests'].values
        coords = data[['Lon', 'Lat']].values
        
        automl_settings = {
            "time_budget": 180,
            "estimator_list": ['rf', 'xgboost', 'xgb_limitdepth', 'extra_tree'],
            "task": 'classification',
            "metric": 'accuracy',
            "seed": 42,
            "verbose": 0,
        }
        
        model = GALAX(
            coords=coords,
            y=y,
            X=X,
            bw='isa',
            kernel='bisquare',
            automl_settings=automl_settings,
            task='classification',
            n_jobs=48,
            x_vars=x_vars
        )
    
    results = model.fit()
    results.summary()
    results.save_results('../results/GALAX_311_class.joblib')

if __name__ == "__main__":
    main()