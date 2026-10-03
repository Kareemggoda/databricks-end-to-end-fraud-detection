# Databricks notebook source
# 01 — Load and Bronze
# Loads the existing PaySim dataset already available in Databricks (loaded via the Kaggle API)

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


# spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{catalog}`.`{schema}`")


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


#  Write Bronze

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


# Continue to **02_Silver_Cleaning**.
