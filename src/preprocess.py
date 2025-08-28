import pandas as pd
import os

# Corrected path to point to the data_processed subdirectory
DATA_DIR = os.path.join("data", "data_processed")

def merge_datasets():
    # Now this path will be correct
    covid = pd.read_csv(os.path.join(DATA_DIR, "processed_owid_data.csv"))
    flu = pd.read_csv(os.path.join(DATA_DIR, "processed_flunet_data.csv"))

    covid["date"] = pd.to_datetime(covid["date"])
    flu["STARTDATE"] = pd.to_datetime(flu["ISO_WEEKSTARTDATE"])

    # Example: Merge on closest date (weekly flu vs daily covid)
    merged = pd.merge_asof(
        covid.sort_values("date"),
        flu.sort_values("STARTDATE"),
        left_on="date", right_on="STARTDATE",
        by=None, tolerance=pd.Timedelta("7D"), direction="nearest"
    )

    # Save the merged file back to the data_processed directory
    merged.to_csv(os.path.join(DATA_DIR, "merged.csv"), index=False)
    print(f"Merged dataset saved: {merged.shape}")

if __name__ == "__main__":
    merge_datasets()