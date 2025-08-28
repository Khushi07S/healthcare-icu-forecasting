 # Healthcare ICU & Disease Forecasting

**Overview**: Lightweight end-to-end project — download OWID COVID and WHO FluNet data → preprocess → short-term ICU forecast → small Spark step → Streamlit dashboard.

## Quick Commands
1. `python -m venv venv`
2. `venv\Scripts\Activate.ps1` (PowerShell) or `venv\Scripts\activate.bat` (cmd)
3. `pip install -r requirements.txt`
4. `python scripts/01_ingest.py`

## Folder structure
- `data\data_raw/` - raw downloaded files
- `data\processed/` - cleaned and processed files
- `scripts/` - ingestion, preprocess, model, spark, dashboard scripts
- `docs/` - diagrams, notes
- `requirements.txt` - Python deps

