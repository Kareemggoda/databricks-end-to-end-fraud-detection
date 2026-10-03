# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Fraud Detection Model
# MAGIC Trains a Gradient-Boosted Trees classifier on the Gold star schema to predict `isFraud`,
# MAGIC using PySpark MLlib (works at full PaySim scale without pulling data to the driver).

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("catalog", "workspace", "Catalog")
dbutils.widgets.text("schema", "paysim", "Schema")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

fact_table = f"{catalog}.{schema}.FactTransactions"
type_table = f"{catalog}.{schema}.DimTransactionType"
date_table = f"{catalog}.{schema}.DimDate"

print("Fact:", fact_table)

# COMMAND ----------

from pyspark.sql import functions as F

fact = spark.table(fact_table)
dim_type = spark.table(type_table).select("transaction_type_key", "transaction_type")
dim_date = spark.table(date_table).select("date_key", "transaction_hour")

df = (
    fact
    .join(dim_type, on="transaction_type_key", how="left")
    .join(dim_date, on="date_key", how="left")
)

print(f"Rows: {df.count():,}")
display(df.groupBy("isFraud").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature selection
# MAGIC - **Excluded:** `nameOrig` / `nameDest` (identifiers, not features), `isFlaggedFraud` (a rule-based
# MAGIC   flag from the simulator — using it would leak the label), and `step` (used only for the split).
# MAGIC - **Used:** transaction type, amounts, balances, `transaction_hour`, and the Silver engineered features.
# MAGIC
# MAGIC **Note on the balance-error features:** `error_balance_orig` / `error_balance_dest` are known to be
# MAGIC extremely predictive in PaySim. This is an artifact of how the simulator records balances for
# MAGIC fraudulent transactions, so very high scores should be read with that in mind — real-world
# MAGIC fraud data rarely contains such a clean signal. Fraud in PaySim also only occurs in `TRANSFER`
# MAGIC and `CASH_OUT` transactions.

# COMMAND ----------

categorical_cols = ["transaction_type"]
numeric_cols = [
    "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "orig_balance_delta", "dest_balance_delta",
    "amount_to_oldbalance_ratio",
    "error_balance_orig", "error_balance_dest",
    "transaction_hour",
]
boolean_cols = ["orig_balance_emptied", "dest_balance_was_zero", "is_merchant_dest"]

label_col = "isFraud"

model_df = df.select("step", *categorical_cols, *numeric_cols, *boolean_cols, label_col)

for c in boolean_cols:
    model_df = model_df.withColumn(c, F.col(c).cast("int"))

model_df = model_df.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train / test split (time-based)
# MAGIC Train on the earliest ~80% of simulated time and test on the latest ~20%. This mirrors real use:
# MAGIC a fraud model is trained on the past and scored on future transactions, and it avoids the
# MAGIC optimistic results a random split can give.

# COMMAND ----------

cutoff_step = model_df.approxQuantile("step", [0.8], 0.001)[0]
print(f"Cutoff step: {cutoff_step}")

train_df = model_df.filter(F.col("step") <= cutoff_step)
test_df = model_df.filter(F.col("step") > cutoff_step)

split_summary = (
    model_df
    .withColumn("split", F.when(F.col("step") <= cutoff_step, "train").otherwise("test"))
    .groupBy("split")
    .agg(
        F.count("*").alias("rows"),
        F.sum(label_col).alias("fraud_rows"),
        F.round(F.avg(label_col) * 100, 4).alias("fraud_rate_pct"),
    )
)
display(split_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline
# MAGIC String-index + one-hot encode the transaction type, assemble all features into a vector, and train
# MAGIC a Gradient-Boosted Trees classifier. GBTs handle imbalanced, non-linear tabular data well, and as a
# MAGIC tree model they need no feature scaling and are robust to outliers.
# MAGIC
# MAGIC Class imbalance is handled with a per-row **weight column** (simpler than SMOTE/resampling),
# MAGIC so the rare fraud class isn't drowned out.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import GBTClassifier

counts = {r[label_col]: r["count"] for r in train_df.groupBy(label_col).count().collect()}
fraud_ratio = counts.get(1, 0) / (counts.get(0, 0) + counts.get(1, 0))
weight_fraud = 1 - fraud_ratio
weight_legit = fraud_ratio
print(f"Train fraud ratio: {fraud_ratio:.5f} | weight fraud={weight_fraud:.5f}, legit={weight_legit:.5f}")

train_weighted = train_df.withColumn(
    "weight",
    F.when(F.col(label_col) == 1, F.lit(weight_fraud)).otherwise(F.lit(weight_legit)),
)

type_indexer = StringIndexer(inputCol="transaction_type", outputCol="type_index", handleInvalid="keep")
type_encoder = OneHotEncoder(inputCol="type_index", outputCol="type_ohe")

assembler = VectorAssembler(
    inputCols=["type_ohe"] + numeric_cols + boolean_cols,
    outputCol="features",
    handleInvalid="keep",
)

gbt = GBTClassifier(
    labelCol=label_col,
    featuresCol="features",
    weightCol="weight",
    maxIter=50,
    maxDepth=5,
    seed=42,
)

pipeline = Pipeline(stages=[type_indexer, type_encoder, assembler, gbt])

# COMMAND ----------

# MAGIC %md ## Train

# COMMAND ----------

model = pipeline.fit(train_weighted)
print("Model trained.")

# COMMAND ----------

# MAGIC %md ## Predict on test set

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

predictions = (
    model.transform(test_df)
    .withColumn("fraud_probability", vector_to_array("probability")[1])
)
display(predictions.select(label_col, "prediction", "fraud_probability").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC Precision, Recall, F1, ROC-AUC, PR-AUC, and a confusion matrix — all for the **fraud class**.
# MAGIC Accuracy is intentionally not reported: a model that predicts "not fraud" every time would score
# MAGIC ~99.9% accuracy and catch nothing. PR-AUC is the most informative single number for rare-event
# MAGIC problems like this.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

roc_auc = BinaryClassificationEvaluator(
    labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
).evaluate(predictions)

pr_auc = BinaryClassificationEvaluator(
    labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
).evaluate(predictions)

cm = {
    (int(r[label_col]), int(r["prediction"])): r["count"]
    for r in predictions.groupBy(label_col, "prediction").count().collect()
}
tn = cm.get((0, 0), 0)
fp = cm.get((0, 1), 0)
fn = cm.get((1, 0), 0)
tp = cm.get((1, 1), 0)

fraud_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
fraud_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
fraud_f1 = (
    2 * fraud_precision * fraud_recall / (fraud_precision + fraud_recall)
    if (fraud_precision + fraud_recall) > 0 else 0.0
)

print("=== Fraud class (isFraud = 1) metrics, threshold 0.5 ===")
print(f"Precision: {fraud_precision:.4f}")
print(f"Recall:    {fraud_recall:.4f}")
print(f"F1-score:  {fraud_f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\n=== Confusion matrix ===")
print(f"                Predicted Legit   Predicted Fraud")
print(f"Actual Legit    {tn:>15,}   {fp:>15,}")
print(f"Actual Fraud    {fn:>15,}   {tp:>15,}")

# COMMAND ----------

confusion_df = spark.createDataFrame(
    [
        ("Actual Legit", "Predicted Legit", tn),
        ("Actual Legit", "Predicted Fraud", fp),
        ("Actual Fraud", "Predicted Legit", fn),
        ("Actual Fraud", "Predicted Fraud", tp),
    ],
    ["actual", "predicted", "count"],
)
display(confusion_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Precision / Recall at different thresholds
# MAGIC Class weighting pushes the model toward catching fraud (high recall), often at the cost of more
# MAGIC false alarms. Raising the decision threshold trades some recall for better precision.
# MAGIC In practice, the business chooses the threshold based on the cost of a missed fraud vs.
# MAGIC the cost of reviewing a false alarm.

# COMMAND ----------

thresholds = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

agg_exprs = []
for t in thresholds:
    pred = F.col("fraud_probability") >= t
    actual = F.col(label_col) == 1
    agg_exprs += [
        F.sum((pred & actual).cast("int")).alias(f"tp_{t}"),
        F.sum((pred & ~actual).cast("int")).alias(f"fp_{t}"),
        F.sum((~pred & actual).cast("int")).alias(f"fn_{t}"),
    ]

row = predictions.agg(*agg_exprs).collect()[0]

threshold_rows = []
for t in thresholds:
    tp_t, fp_t, fn_t = row[f"tp_{t}"], row[f"fp_{t}"], row[f"fn_{t}"]
    p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    threshold_rows.append((float(t), int(tp_t), int(fp_t), int(fn_t), round(p, 4), round(r, 4), round(f1, 4)))

display(spark.createDataFrame(
    threshold_rows,
    ["threshold", "true_pos", "false_pos", "false_neg", "precision", "recall", "f1"],
))

# COMMAND ----------

# MAGIC %md ## Feature importance

# COMMAND ----------

import pandas as pd

gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()

try:
    attrs = predictions.schema["features"].metadata["ml_attr"]["attrs"]
    idx_to_name = {a["idx"]: a["name"] for group in attrs.values() for a in group}
except Exception as e:
    print(f"Could not read feature metadata ({e}); falling back to generic names.")
    idx_to_name = {}

imp_df = pd.DataFrame(
    [(idx_to_name.get(i, f"feature_{i}"), float(v)) for i, v in enumerate(importances)],
    columns=["feature", "importance"],
).sort_values("importance", ascending=False)

display(spark.createDataFrame(imp_df))

# COMMAND ----------

# MAGIC %md ## Save the model
# MAGIC Saved to a Unity Catalog Volume so it persists and can be reloaded with `PipelineModel.load(path)`.

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.models")
model_path = f"/Volumes/{catalog}/{schema}/models/fraud_gbt_model"

model.write().overwrite().save(model_path)
print(f"Model saved to: {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC A single GBTClassifier trained on Gold-layer features (excluding `nameOrig`/`nameDest` and
# MAGIC `isFlaggedFraud`), with class weighting for imbalance and a time-based train/test split.
# MAGIC Evaluated on fraud-class Precision, Recall, F1, ROC-AUC, PR-AUC, and a confusion matrix, with a
# MAGIC threshold table showing the precision/recall trade-off.
# MAGIC
# MAGIC This closes the loop: **raw data → Bronze → Silver → Gold star schema → dashboard + fraud model.**
