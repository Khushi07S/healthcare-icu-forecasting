import pandas as pd
import os

RAW_DATA_PATH = "healthcare-icu-forecasting\data\data_raw"
PROCESSED_DATA_PATH = "healthcare-icu-forecasting\data\data_processed"

def load_owid_data():
    """Loads the OWID COVID-19 data."""
    filepath = os.path.join(RAW_DATA_PATH, "owid-covid-data.csv")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    return pd.read_csv(filepath)

def load_flunet_data():
    """Loads the FLUNET data."""
    filepath = os.path.join(RAW_DATA_PATH, "Flunet.csv")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    return pd.read_csv(filepath)

def save_processed_data(df, filename):
    """Saves a DataFrame to the processed data directory."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    filepath = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Processed file saved: {filepath}")

def main():
    """Main function to orchestrate the data processing pipeline."""
    
    # 1. Load the raw data
    print("Loading raw data...")
    owid_df = load_owid_data()
    flunet_df = load_flunet_data()

    # Simple check to ensure data was loaded
    if not owid_df.empty:
        print(f"✅ OWID data loaded. Shape: {owid_df.shape}")
        # 2. Perform some processing (example only)
        # For a real project, you'd add more complex processing here
        processed_owid_df = owid_df.head(182275)
        
        # 3. Save the processed data
        save_processed_data(processed_owid_df, "processed_owid_data.csv")
    
    if not flunet_df.empty:
        print(f"✅ FLUNET data loaded. Shape: {flunet_df.shape}")
        # Perform some processing (example only)
        processed_flunet_df = flunet_df.head(429436)
        
        # Save the processed data
        save_processed_data(processed_flunet_df, "processed_flunet_data.csv")
        
    print("\n🏁 Data ingestion complete.")
    print("Raw files available in:", RAW_DATA_PATH)
    for f in os.listdir(RAW_DATA_PATH):
        print(" -", f)
        
    print("\nProcessed files available in:", PROCESSED_DATA_PATH)
    for f in os.listdir(PROCESSED_DATA_PATH):
        print(" -", f)

if __name__ == "__main__":
    main()