from pathlib import Path

# Root project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Raw datasets
IHSG_FILE = DATA_RAW / "historis_ihsg.csv"
KURS_FILE = DATA_RAW / "nilai_tukar.csv"
BI_RATE_FILE = DATA_RAW / "bi_rate_monthly.csv"
INFLASI_FILE = DATA_RAW / "inflasi_monthly.csv"
PDB_FILE = DATA_RAW / "pdb_monthly.csv"
