-- ============================================================
-- PaySim Fraud Dashboard — Databricks Lakeview (AI/BI Dashboard)
-- Source: workspace.paysim Gold star schema
-- ============================================================
--
-- Design: ONE pre-aggregated dataset powers all 8 tiles.
-- Grain = one row per (hour × transaction type × fraud status), max ~7,400 rows.
-- Because every tile uses the same dataset, all three dashboard filters
-- (Transaction Type, Date/Time, Fraud Status) apply to every tile, and the
-- dashboard stays fast without scanning 6.3M rows per tile.
--
-- In the dashboard's Data tab, create a dataset named: fraud_summary
-- and paste the query below.
-- ============================================================

SELECT
  d.calendar_datetime,
  d.calendar_date,
  d.transaction_day,
  d.transaction_hour,
  t.transaction_type,
  df.fraud_status,                         -- 'Fraud' / 'Non-Fraud'
  df.fraud_label,                          -- detailed label incl. flagged status
  COUNT(*)                                                    AS txn_count,
  SUM(f.amount)                                               AS total_amount,
  SUM(f.isFraud)                                              AS fraud_count,
  SUM(CASE WHEN f.isFraud = 1 THEN f.amount ELSE 0 END)       AS fraud_amount
FROM workspace.paysim.FactTransactions f
JOIN workspace.paysim.DimDate            d  ON f.date_key             = d.date_key
JOIN workspace.paysim.DimTransactionType t  ON f.transaction_type_key = t.transaction_type_key
JOIN workspace.paysim.DimFraud           df ON f.fraud_key            = df.fraud_key
GROUP BY
  d.calendar_datetime, d.calendar_date, d.transaction_day, d.transaction_hour,
  t.transaction_type, df.fraud_status, df.fraud_label;


-- ============================================================
-- Tile definitions (all built on dataset: fraud_summary)
-- ============================================================
--
-- 1. Total Transactions         Counter   SUM(txn_count)
-- 2. Total Transaction Amount   Counter   SUM(total_amount)
-- 3. Fraudulent Transactions    Counter   SUM(fraud_count)
-- 4. Fraud Rate (%)             Counter   custom calculation:
--                                         SUM(fraud_count) * 100.0 / SUM(txn_count)
-- 5. Transactions by Type       Bar       X: transaction_type   Y: SUM(txn_count)
-- 6. Fraud by Type              Bar       X: transaction_type   Y: SUM(fraud_count)
-- 7. Fraud over Time            Line      X: calendar_datetime (by hour or day)
--                                         Y: SUM(fraud_count)
-- 8. Fraud vs Non-Fraud         Pie/Bar   Group: fraud_status   Value: SUM(txn_count)
--
-- Filters (dashboard filter widgets, applied to fraud_summary):
--   Transaction Type   Multiple-value dropdown  -> transaction_type
--   Date/Time          Date range picker        -> calendar_datetime
--   Fraud Status       Single-value dropdown    -> fraud_status
--
-- Note: with Fraud Status = 'Fraud' selected, Fraud Rate shows 100% — that is
-- expected behaviour, since the filter removes all non-fraud rows.


-- ============================================================
-- Validation query (run once in the SQL editor, not a dashboard tile)
-- Totals from the summary dataset must match the fact table exactly.
-- Expected: 6,362,620 transactions, 8,213 fraud.
-- ============================================================

SELECT
  (SELECT COUNT(*)     FROM workspace.paysim.FactTransactions) AS fact_txn_count,
  (SELECT SUM(isFraud) FROM workspace.paysim.FactTransactions) AS fact_fraud_count,
  SUM(txn_count)   AS summary_txn_count,
  SUM(fraud_count) AS summary_fraud_count
FROM (
  SELECT COUNT(*) AS txn_count, SUM(f.isFraud) AS fraud_count
  FROM workspace.paysim.FactTransactions f
  JOIN workspace.paysim.DimDate            d  ON f.date_key             = d.date_key
  JOIN workspace.paysim.DimTransactionType t  ON f.transaction_type_key = t.transaction_type_key
  JOIN workspace.paysim.DimFraud           df ON f.fraud_key            = df.fraud_key
  GROUP BY d.calendar_datetime, t.transaction_type, df.fraud_status, df.fraud_label
);
