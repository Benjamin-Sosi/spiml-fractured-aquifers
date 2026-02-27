from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def train_models(features, target):
    """
    Train:
    1. Global black-box model
    2. Structure-aware regime-conditioned model
    """

    # --- Global model ---
    global_model = RandomForestRegressor(random_state=0)
    global_model.fit(features, target)
    global_pred = global_model.predict(features)
    global_rmse = np.sqrt(mean_squared_error(target, global_pred))

    # --- Structure-aware model ---
    fracture_mask = features["regime"] == 1
    matrix_mask = features["regime"] == 0

    model_fracture = RandomForestRegressor(random_state=0)
    model_matrix = RandomForestRegressor(random_state=0)

    model_fracture.fit(features[fracture_mask], target[fracture_mask])
    model_matrix.fit(features[matrix_mask], target[matrix_mask])

    pred_fracture = model_fracture.predict(features[fracture_mask])
    pred_matrix = model_matrix.predict(features[matrix_mask])

    structure_pred = np.zeros(len(target))
    structure_pred[fracture_mask] = pred_fracture
    structure_pred[matrix_mask] = pred_matrix

    structure_rmse = np.sqrt(mean_squared_error(target, structure_pred))

    return {
        "global_rmse": global_rmse,
        "structure_rmse": structure_rmse,
        "predictions": structure_pred
    }
