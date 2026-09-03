from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
BRONZE_DIR = DATA / "bronze" / "complaints"
SILVER_DIR = DATA / "silver" / "complaints"
GOLD_DIR = DATA / "gold"
SILVER_PATH = SILVER_DIR / "part-0000.parquet"
MODELS_DIR = ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "product_classifier.joblib"
TIMELY_CLASSIFIER_PATH = MODELS_DIR / "timely_classifier.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
TIMELY_METRICS_PATH = MODELS_DIR / "metrics_timely.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
INDEX_PATH = MODELS_DIR / "complaint_index.joblib"
MLFLOW_DB = ROOT / "mlflow.db"
QUALITY_REPORT_PATH = MODELS_DIR / "quality_report.json"
SQL_QUERIES = ROOT / "sql" / "02_analytical_queries.sql"
SCHEMA_SQL = ROOT / "sql" / "01_star_schema.sql"
GOLD_TABLES = (
    "fact_complaints",
    "dim_company",
    "dim_product",
    "dim_issue",
    "dim_geo",
    "dim_channel",
    "dim_date",
)
