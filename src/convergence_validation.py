import numpy as np


def validate_convergence(results):
    """
    Convergence-based validation:
    Checks whether structure-aware RMSE improves over global model.
    """

    global_rmse = results["global_rmse"]
    structure_rmse = results["structure_rmse"]

    print("\n--- Model Performance ---")
    print(f"Global Model RMSE: {global_rmse:.2f}")
    print(f"Structure-Aware RMSE: {structure_rmse:.2f}")

    if structure_rmse < global_rmse:
        print("Convergence validation PASSED: Structural conditioning improves inference.")
    else:
        print("Convergence validation FAILED: No structural improvement detected.")
