import pandas as pd
import os

def main():
    input_path = "data/raw/notices.csv"
    output_path = "data/processed/tenders_labeled.csv"

    # Load raw data
    df = pd.read_csv(input_path)

    # Drop rows with missing cost data
    df = df.dropna(subset=["Value Low", "Value High"])

    # Convert to numeric (remove scientific notation)
    df["Value High"] = pd.to_numeric(df["Value High"], errors="coerce").astype(float).astype(int)
    df["Value Low"] = pd.to_numeric(df["Value Low"], errors="coerce").astype(float).astype(int)

    # Create final columns
    df["estimated_cost"] = df["Value Low"]
    df["final_cost"] = df["Value High"]
    df["overshot"] = (df["final_cost"] > df["estimated_cost"]).astype(int)

    # Keep only needed columns
    df_cleaned = df[[
        "Title", "Description", "Region", "Organisation Name",
        "estimated_cost", "final_cost", "overshot"
    ]]

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned dataset
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")


if __name__ == "__main__":
    main()
