# Databricks notebook source
# 02 — Silver Cleaning
# Reads Bronze, fixes types, handles nulls/duplicates, validates transactions,
# adds a small set of useful engineered features.

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "paysim", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

bronze_table = f"{catalog}.{schema}.bronze_paysim"
silver_table = f"{catalog}.{schema}.silver_paysim"

print("Bronze source:", bronze_table)
print("Silver target:", silver_table)


from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType, BooleanType

df = spark.table(bronze_table)
print(f"Bronze row count: {df.count():,}")


# Fix data types
# PaySim raw columns: `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`

df_typed = (
    df
    .withColumn("step", F.col("step").cast(IntegerType()))
    .withColumn("type", F.col("type").cast(StringType()))
    .withColumn("amount", F.col("amount").cast(DoubleType()))
    .withColumn("nameOrig", F.col("nameOrig").cast(StringType()))
    .withColumn("oldbalanceOrg", F.col("oldbalanceOrg").cast(DoubleType()))
    .withColumn("newbalanceOrig", F.col("newbalanceOrig").cast(DoubleType()))
    .withColumn("nameDest", F.col("nameDest").cast(StringType()))
    .withColumn("oldbalanceDest", F.col("oldbalanceDest").cast(DoubleType()))
    .withColumn("newbalanceDest", F.col("newbalanceDest").cast(DoubleType()))
    .withColumn("isFraud", F.col("isFraud").cast(IntegerType()))
    .withColumn("isFlaggedFraud", F.col("isFlaggedFraud").cast(IntegerType()))
)


# Handle nulls
# Drop rows missing any field essential to the transaction; PaySim is normally complete, so this is a safety net.


required_cols = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]

null_report = df_typed.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in required_cols]
)
display(null_report)

df_clean = df_typed.dropna(subset=required_cols)
print(f"Rows after null drop: {df_clean.count():,}")


# Remove duplicates
# A transaction is treated as duplicate if every business column matches exactly.


before = df_clean.count()
df_clean = df_clean.dropDuplicates(required_cols + ["isFlaggedFraud"])
after = df_clean.count()
print(f"Removed {before - after:,} duplicate rows")


# Validate transaction data
# Flag (don't silently drop) rows with impossible values, then filter out the clearly invalid ones

df_validated = (
    df_clean
    .withColumn("is_negative_amount", F.col("amount") < 0)
    .withColumn("is_negative_balance",
                (F.col("oldbalanceOrg") < 0) | (F.col("newbalanceOrig") < 0) |
                (F.col("oldbalanceDest") < 0) | (F.col("newbalanceDest") < 0))
    .withColumn("is_valid_type",
                F.col("type").isin("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"))
)

invalid_count = df_validated.filter(
    F.col("is_negative_amount") | F.col("is_negative_balance") | (~F.col("is_valid_type"))
).count()
print(f"Invalid rows found: {invalid_count:,}")

df_validated = (
    df_validated
    .filter(~F.col("is_negative_amount"))
    .filter(~F.col("is_negative_balance"))
    .filter(F.col("is_valid_type"))
    .drop("is_negative_amount", "is_negative_balance", "is_valid_type")
)

print(f"Rows after validation: {df_validated.count():,}")


# Feature engineering
# **transaction_hour**: hour of day (PaySim `step` = 1 simulated hour; 744 steps = 31 days)
# **balance change features**: how balances actually moved
# **amount-related features**: ratio of amount to sender's balance, whether amount drains the account
# **basic fraud-related indicators**: the classic PaySim balance-error signals, which are strongly
# associated with fraud in this dataset (fraudulent transfers/cash-outs tend to zero out balances in ways that don't reconcile).


df_features = (
    df_validated
    # time features
    .withColumn("transaction_day", (F.col("step") / F.lit(24)).cast(IntegerType()) + 1)
    .withColumn("transaction_hour", F.col("step") % F.lit(24))

    # balance change features
    .withColumn("orig_balance_delta", F.col("newbalanceOrig") - F.col("oldbalanceOrg"))
    .withColumn("dest_balance_delta", F.col("newbalanceDest") - F.col("oldbalanceDest"))

    # amount-related features
    .withColumn(
        "amount_to_oldbalance_ratio",
        F.when(F.col("oldbalanceOrg") > 0, F.col("amount") / F.col("oldbalanceOrg")).otherwise(F.lit(0.0)),
    )
    .withColumn("orig_balance_emptied", (F.col("newbalanceOrig") == 0) & (F.col("oldbalanceOrg") > 0))
    .withColumn("dest_balance_was_zero", F.col("oldbalanceDest") == 0)

    # basic fraud-related indicators (balance reconciliation errors — a known strong PaySim signal)
    .withColumn(
        "error_balance_orig",
        F.col("newbalanceOrig") + F.col("amount") - F.col("oldbalanceOrg"),
    )
    .withColumn(
        "error_balance_dest",
        F.col("oldbalanceDest") + F.col("amount") - F.col("newbalanceDest"),
    )
    .withColumn(
        "is_merchant_dest",
        F.col("nameDest").startswith("M"),
    )
)

display(df_features.limit(10))

# | Feature                      | Simple meaning                                          |
# | ---------------------------- | ------------------------------------------------------- |
# | `orig_balance_delta`         | How much the sender's balance changed                   |
# | `dest_balance_delta`         | How much the receiver's balance changed                 |
# | `amount_to_oldbalance_ratio` | How much of the sender's old balance was sent           |
# | `orig_balance_emptied`       | Did the sender's balance become zero?                   |
# | `dest_balance_was_zero`      | Was the receiver's balance zero before the transaction? |
# | `error_balance_orig`         | Is the sender's balance calculation correct?            |
# | `error_balance_dest`         | Is the receiver's balance calculation correct?          |
# | `is_merchant_dest`           | Is the receiver a merchant?                             |

# Write Silver

(
    df_features.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)

print(f"Silver table written: {silver_table}")
print(f"Row count: {spark.table(silver_table).count():,}")

# Silver table: `silver_paysim` — typed, deduplicated, validated, feature-enriched.
# Continue to **03_Gold_Data_Model**.
