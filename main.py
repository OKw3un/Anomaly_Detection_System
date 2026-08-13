import pandas as pd

from src.profiler.profiler import DataProfiler
from src.profiler.router import ModelRouter

def main():

    import os
    
    # =========================================================
    # 1. Load Dataset
    # =========================================================

    file_path = "data/timeseries/TimeSeries.csv"
    df = pd.read_csv(file_path)

    print(f"Dataset loaded: {df.shape}")

    # =========================================================
    # 2. Profile Dataset
    # =========================================================

    profiler = DataProfiler(df, data_dir=os.path.dirname(file_path))

    meta_vector = profiler.profile()

    print("\n--- Meta-Feature Vector ---")

    print(
        meta_vector.to_dict()
    )

    # =========================================================
    # 3. Model Routing
    # =========================================================

    router = ModelRouter()

    config = router.route(
        meta_vector
    )

    print("\n--- Model Routing Output ---")

    print(config)

    # =========================================================
    # 4. Important Summary
    # =========================================================

    print("\n--- Dataset Summary ---")

    print(
        f"Dataset Type: "
        f"{meta_vector.dataset_type}"
    )

    print(
        f"Rows: "
        f"{meta_vector.total_rows}"
    )

    print(
        f"Columns: "
        f"{meta_vector.total_columns}"
    )

    print(
        f"Continuous Columns: "
        f"{len(meta_vector.continuous_cols)}"
    )

    print(
        f"Categorical Columns: "
        f"{len(meta_vector.categorical_cols)}"
    )

    print(
        f"ID Columns: "
        f"{meta_vector.id_cols}"
    )

    print(
        f"Label Columns: "
        f"{meta_vector.label_cols}"
    )

    print(
        f"Temporal Candidates: "
        f"{meta_vector.temporal_candidate_cols}"
    )

    print(
        f"High Dimensionality: "
        f"{meta_vector.high_dimensionality}"
    )

    print(
        f"Missing Ratio: "
        f"{meta_vector.missing_data_ratio:.4f}"
    )

if __name__ == "__main__":
    main()

