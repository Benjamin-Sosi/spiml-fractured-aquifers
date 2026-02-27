from src.hydrostructure import generate_structural_data
from src.feature_engineering import engineer_features
from src.structure_aware_model import train_models
from src.convergence_validation import validate_convergence


def main():
    print("Running SPIML synthetic workflow...\n")

    data = generate_structural_data()
    features, target = engineer_features(data)
    results = train_models(features, target)
    validate_convergence(results)


if __name__ == "__main__":
    main()
