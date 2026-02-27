import pandas as pd


def engineer_features(data: pd.DataFrame):
    """
    Physics-consistent feature engineering.
    Distance to fracture centerline used as structural proxy.
    """

    data = data.copy()

    # Fracture center at x = 50
    data["dist_to_fracture"] = abs(data["x"] - 50)

    features = data[["x", "y", "dist_to_fracture", "regime"]]
    target = data["transmissivity"]

    return features, target
