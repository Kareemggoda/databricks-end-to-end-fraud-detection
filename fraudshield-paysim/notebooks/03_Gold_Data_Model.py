# Databricks notebook source
# # 03 — Gold Data Model (Star Schema)
# Builds a simple star schema on top of Silver:
#                     DimDate
#                        │
#  DimTransactionType ── FactTransactions ── DimFraud


dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "paysim", "Schema")
dbutils.widgets.text("sim_start_date", "2024-01-01", "Simulation start date (arbitrary anchor for DimDate)")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
sim_start_date = dbutils.widgets.get("sim_start_date")

silver_table = f"{catalog}.{schema}.silver_paysim"

dim_date_table = f"{catalog}.{schema}.DimDate"
dim_type_table = f"{catalog}.{schema}.DimTransactionType"
dim_fraud_table = f"{catalog}.{schema}.DimFraud"
fact_table = f"{catalog}.{schema}.FactTransactions"

print("Silver source:", silver_table)
print("Gold targets:", dim_date_table, dim_type_table, dim_fraud_table, fact_table, sep="\n  ")


# Drop existing keys
# Primary/foreign keys are removed before the tables are rewritten (a foreign key can block
# overwriting the dimension it references). They are re-created at the end of the notebook.


constraints_to_drop = {
    "FactTransactions": ["fk_fact_date", "fk_fact_type", "fk_fact_fraud"],
    "DimDate": ["pk_dimdate"],
    "DimTransactionType": ["pk_dimtype"],
    "DimFraud": ["pk_dimfraud"],
}
for table, names in constraints_to_drop.items():
    for name in names:
        try:
            spark.sql(f"ALTER TABLE {catalog}.{schema}.{table} DROP CONSTRAINT IF EXISTS {name}")
        except Exception:
            pass  # table doesn't exist yet on the first run 
            # This is the first time running the notebook, so the Gold tables may not have been created yet. Therefore, there are no constraints to remove.

print("Old keys dropped (if any).")


from pyspark.sql import functions as F
from pyspark.sql.window import Window

silver = spark.table(silver_table)
silver_count = silver.count()
print(f"Silver row count: {silver_count:,}")


# DimDate
# PaySim's `step` is simulated hours (1 hour per step, 744 steps ≈ 31 days). There's no real calendar
# date in the raw data, so DimDate anchors `transaction_day` to an arbitrary start date to make the
# dimension usable for date filtering in the dashboard.
# Grain: one row per simulated hour. `calendar_datetime` supports hourly "Fraud over Time" charts.


dim_date = (
    silver
    .select("transaction_day", "transaction_hour")
    .distinct()
    .withColumn(
        "calendar_date",
        F.expr(f"date_add(to_date('{sim_start_date}'), transaction_day - 1)"),
    )
    .withColumn(
        "calendar_datetime",
        F.expr("timestampadd(HOUR, transaction_hour, cast(calendar_date as timestamp))"),
    )
    .withColumn("date_key", (F.col("transaction_day") * 100 + F.col("transaction_hour")).cast("int"))
    .withColumn("day_of_week", F.date_format("calendar_date", "EEEE"))
    .withColumn("month", F.month("calendar_date"))
    .withColumn("year", F.year("calendar_date"))
    .select(
        "date_key", "transaction_day", "transaction_hour",
        "calendar_date", "calendar_datetime", "day_of_week", "month", "year",
    )
)

dim_date.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(dim_date_table)
print(f"{dim_date_table}: {spark.table(dim_date_table).count():,} rows")
display(spark.table(dim_date_table).orderBy("date_key").limit(5))


# DimTransactionType


dim_type = (
    silver
    .select("type")
    .distinct()
    .withColumn("transaction_type_key", F.row_number().over(Window.orderBy("type")))
    .withColumnRenamed("type", "transaction_type")
    .select("transaction_type_key", "transaction_type")
)

dim_type.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(dim_type_table)
print(f"{dim_type_table}: {spark.table(dim_type_table).count():,} rows")
display(spark.table(dim_type_table).orderBy("transaction_type_key"))


# DimFraud
# One row per distinct combination of `isFraud` / `isFlaggedFraud`, with a readable label.


dim_fraud = (
    silver
    .select("isFraud", "isFlaggedFraud")
    .distinct()
    .withColumn("fraud_key", F.row_number().over(Window.orderBy("isFraud", "isFlaggedFraud")))
    .withColumn(
        "fraud_label",
        F.when((F.col("isFraud") == 1) & (F.col("isFlaggedFraud") == 1), "Fraud - Flagged")
         .when((F.col("isFraud") == 1) & (F.col("isFlaggedFraud") == 0), "Fraud - Not Flagged")
         .when((F.col("isFraud") == 0) & (F.col("isFlaggedFraud") == 1), "Legit - Flagged (false positive)")
         .otherwise("Legit"),
    )
    .withColumn("fraud_status", F.when(F.col("isFraud") == 1, "Fraud").otherwise("Non-Fraud"))
    .select("fraud_key", "isFraud", "isFlaggedFraud", "fraud_label", "fraud_status")
)

dim_fraud.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(dim_fraud_table)
print(f"{dim_fraud_table}: {spark.table(dim_fraud_table).count():,} rows")
display(spark.table(dim_fraud_table).orderBy("fraud_key"))


# FactTransactions
# Transaction-level grain, one row per PaySim transaction, joined to the dimension keys plus the
# measures and engineered features needed for the dashboard and the ML step.
# Dimensions are re-read from their Delta tables so the fact keys always match what was written.


dim_type_lk = spark.table(dim_type_table).select("transaction_type_key", "transaction_type")
dim_fraud_lk = spark.table(dim_fraud_table).select("fraud_key", "isFraud", "isFlaggedFraud")

fact = (
    silver
    .withColumn("date_key", (F.col("transaction_day") * 100 + F.col("transaction_hour")).cast("int"))
    .join(dim_type_lk, silver["type"] == dim_type_lk["transaction_type"], "left")
    .join(dim_fraud_lk, on=["isFraud", "isFlaggedFraud"], how="left")
    .select(
        F.monotonically_increasing_id().alias("transaction_key"),
        "date_key",
        "transaction_type_key",
        "fraud_key",
        F.col("step"),
        F.col("nameOrig"),
        F.col("nameDest"),
        F.col("amount"),
        F.col("oldbalanceOrg"),
        F.col("newbalanceOrig"),
        F.col("oldbalanceDest"),
        F.col("newbalanceDest"),
        F.col("orig_balance_delta"),
        F.col("dest_balance_delta"),
        F.col("amount_to_oldbalance_ratio"),
        F.col("orig_balance_emptied"),
        F.col("dest_balance_was_zero"),
        F.col("error_balance_orig"),
        F.col("error_balance_dest"),
        F.col("is_merchant_dest"),
        F.col("isFraud"),
        F.col("isFlaggedFraud"),
    )
)

(
    fact.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(fact_table)
)

fact_count = spark.table(fact_table).count()
print(f"{fact_table}: {fact_count:,} rows")
display(spark.table(fact_table).limit(5))


# Quick sanity checks
# Fact row count must equal Silver (joins must not duplicate or drop rows)
# No NULL foreign keys (every fact row must match a dimension row)


assert fact_count == silver_count, f"Row mismatch: Silver={silver_count:,}, Fact={fact_count:,}"

null_keys = spark.sql(f"""
SELECT
  SUM(CASE WHEN date_key IS NULL THEN 1 ELSE 0 END)             AS null_date_key,
  SUM(CASE WHEN transaction_type_key IS NULL THEN 1 ELSE 0 END) AS null_type_key,
  SUM(CASE WHEN fraud_key IS NULL THEN 1 ELSE 0 END)            AS null_fraud_key
FROM {fact_table}
""")
display(null_keys)

orphan_dates = spark.sql(f"""
SELECT COUNT(*) AS orphan_date_keys
FROM {fact_table} f
LEFT ANTI JOIN {dim_date_table} d ON f.date_key = d.date_key
""")
display(orphan_dates)

print("Row count check passed: Fact matches Silver.")


spark.sql(f"""
SELECT
  d.fraud_label,
  COUNT(*) AS txn_count,
  ROUND(SUM(f.amount), 2) AS total_amount
FROM {fact_table} f
JOIN {dim_fraud_table} d ON f.fraud_key = d.fraud_key
GROUP BY d.fraud_label
ORDER BY txn_count DESC
""").display()


# Primary and foreign keys
# Informational (not enforced) Unity Catalog constraints that document the star schema.
# They let Catalog Explorer draw the entity relationship diagram for the Gold tables.


pk_defs = [
    ("DimDate", "date_key", "pk_dimdate"),
    ("DimTransactionType", "transaction_type_key", "pk_dimtype"),
    ("DimFraud", "fraud_key", "pk_dimfraud"),
]
for table, col, name in pk_defs:
    spark.sql(f"ALTER TABLE {catalog}.{schema}.{table} ALTER COLUMN {col} SET NOT NULL")
    spark.sql(f"ALTER TABLE {catalog}.{schema}.{table} ADD CONSTRAINT {name} PRIMARY KEY ({col})")

fk_defs = [
    ("date_key", "DimDate", "fk_fact_date"),
    ("transaction_type_key", "DimTransactionType", "fk_fact_type"),
    ("fraud_key", "DimFraud", "fk_fact_fraud"),
]
for col, dim, name in fk_defs:
    spark.sql(f"""
        ALTER TABLE {catalog}.{schema}.FactTransactions
        ADD CONSTRAINT {name} FOREIGN KEY ({col}) REFERENCES {catalog}.{schema}.{dim}({col})
    """)

print("Primary and foreign keys added.")

display(spark.sql(f"""
SELECT table_name, constraint_name, constraint_type
FROM {catalog}.information_schema.table_constraints
WHERE table_schema = '{schema}'
ORDER BY constraint_type DESC, table_name
"""))


# Gold star schema is ready: `FactTransactions`, `DimDate`, `DimTransactionType`, `DimFraud`,
# with primary and foreign keys declared.
# Use these tables to build the dashboard (see `dashboard_queries.sql`) and continue to
# **04_Fraud_Detection_Model**.
