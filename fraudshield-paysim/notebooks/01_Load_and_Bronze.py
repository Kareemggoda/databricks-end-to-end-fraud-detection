# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Load and Bronze
# MAGIC Loads the existing PaySim dataset already available in Databricks (loaded via the Kaggle API)
# MAGIC and lands it as a Delta **Bronze** table with no transformations beyond adding ingestion metadata.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets — set these to match your environment
# MAGIC - `source_table`: fully qualified name of the existing PaySim table, **or**
# MAGIC - `source_path`: a file path (CSV/Parquet) if the data was landed as files rather than a table.
# MAGIC
# MAGIC Only one of the two needs to resolve — the notebook tries `source_table` first.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "paysim", "Schema")
dbutils.widgets.text("source_table", "workspace.default.transactions", "Existing source table (if any)")
dbutils.widgets.text("source_path", "", "Existing source file path (if source_table is empty/invalid)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
source_table = dbutils.widgets.get("source_table")
source_path = dbutils.widgets.get("source_path")

bronze_table = f"{catalog}.{schema}.bronze_paysim"

# COMMAND ----------

# spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")

# COMMAND ----------

from pyspark.sql import functions as F

df = None

if source_table:
    try:
        df = spark.table(source_table)
        print(f"Loaded existing table: {source_table}")
    except Exception as e:
        print(f"Could not read source_table '{source_table}': {e}")

if df is None and source_path:
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(source_path)
    )
    print(f"Loaded from file path: {source_path}")

if df is None:
    raise ValueError(
        "Could not locate the existing PaySim data. "
        "Set the 'source_table' widget to the table you loaded via the Kaggle API, "
        "or set 'source_path' to the file location."
    )

print(f"Row count: {df.count():,}")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Bronze
# MAGIC Raw data as-is, plus minimal ingestion metadata (no cleaning, no type fixes — that happens in Silver).

# COMMAND ----------

bronze_df = (
    df
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source", F.lit(source_table if source_table else source_path))
)

(
    bronze_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)

print(f"Bronze table written: {bronze_table}")
display(spark.table(bronze_table).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC Bronze table: `bronze_paysim` — raw PaySim rows, unmodified, with ingestion timestamp/source tag.
# MAGIC Continue to **02_Silver_Cleaning**.
