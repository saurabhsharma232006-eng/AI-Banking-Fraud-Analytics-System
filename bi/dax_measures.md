# Power BI DAX Measures Cheat-Sheet
## AI-Powered Banking & Fraud Transaction Analytics

> Paste these measures into Power BI Desktop via **Model View → New Measure**.
> All measures assume `fact_transactions` is the central fact table connected to dimension tables.

---

## 📊 Core Volume Metrics

### Total Transactions
```dax
Total Transactions =
COUNTROWS(fact_transactions)
```

### Total Transaction Amount
```dax
Total Amount =
SUM(fact_transactions[amount])
```

### Unique Accounts
```dax
Unique Accounts =
DISTINCTCOUNT(fact_transactions[account_key])
```

---

## 🚨 Fraud Metrics

### Total Fraud Transactions
```dax
Total Fraud Transactions =
CALCULATE(
    COUNTROWS(fact_transactions),
    fact_transactions[is_fraud] = 1
)
```

### Total Fraud Amount
```dax
Total Fraud Amount =
SUM(fact_transactions[fraud_loss])
```

### Fraud Rate (%)
```dax
Fraud Rate % =
DIVIDE(
    [Total Fraud Transactions],
    [Total Transactions],
    0
) * 100
```

### Average Fraud Transaction Value
```dax
Avg Fraud Transaction Value =
CALCULATE(
    AVERAGE(fact_transactions[amount]),
    fact_transactions[is_fraud] = 1
)
```

### Fraud Loss Ratio
```dax
Fraud Loss Ratio =
DIVIDE(
    [Total Fraud Amount],
    [Total Amount],
    0
)
```

---

## ⚠️ Risk Tier Metrics

### High Risk Transactions
```dax
High Risk Transactions =
CALCULATE(
    COUNTROWS(fact_transactions),
    fact_transactions[risk_tier] IN {"HIGH", "CRITICAL"}
)
```

### Critical Risk Count
```dax
Critical Risk Count =
CALCULATE(
    COUNTROWS(fact_transactions),
    fact_transactions[risk_tier] = "CRITICAL"
)
```

### Average Fraud Probability (Scored)
```dax
Avg Fraud Probability =
CALCULATE(
    AVERAGE(fact_transactions[fraud_probability]),
    NOT(ISBLANK(fact_transactions[fraud_probability]))
)
```

---

## 📈 Time Intelligence

### Fraud Transactions MTD
```dax
Fraud Transactions MTD =
CALCULATE(
    [Total Fraud Transactions],
    DATESMTD(dim_date[date])
)
```

### Fraud Amount YTD
```dax
Fraud Amount YTD =
CALCULATE(
    [Total Fraud Amount],
    DATESYTD(dim_date[date])
)
```

### Fraud Rate MoM Change
```dax
Fraud Rate MoM % Change =
VAR CurrentMonth = [Fraud Rate %]
VAR PriorMonth =
    CALCULATE(
        [Fraud Rate %],
        DATEADD(dim_date[date], -1, MONTH)
    )
RETURN
    DIVIDE(CurrentMonth - PriorMonth, PriorMonth, 0) * 100
```

### Rolling 7-Day Fraud Count
```dax
Rolling 7D Fraud Count =
CALCULATE(
    [Total Fraud Transactions],
    DATESINPERIOD(dim_date[date], LASTDATE(dim_date[date]), -7, DAY)
)
```

---

## 🏪 Merchant Analytics

### Fraud Rate by Merchant
```dax
Fraud Rate by Merchant =
DIVIDE(
    CALCULATE(COUNTROWS(fact_transactions), fact_transactions[is_fraud] = 1),
    COUNTROWS(fact_transactions),
    0
) * 100
```

### Top Merchant Fraud Loss
```dax
Top Merchant Fraud Loss =
MAXX(
    ALLSELECTED(dim_merchants[merchant_name]),
    CALCULATE([Total Fraud Amount])
)
```

---

## 🧑‍💼 Customer Concentration

### High-Risk Customer Concentration
```dax
High Risk Customer Concentration =
DIVIDE(
    CALCULATE(
        DISTINCTCOUNT(fact_transactions[account_key]),
        fact_transactions[risk_tier] IN {"HIGH", "CRITICAL"}
    ),
    [Unique Accounts],
    0
) * 100
```

### Top 10% Account Fraud Contribution
```dax
Top Account Fraud Pct =
VAR TotalFraud = [Total Fraud Transactions]
VAR Top10Threshold =
    PERCENTILE.INC(
        ADDCOLUMNS(
            SUMMARIZE(fact_transactions, fact_transactions[account_key]),
            "FraudCnt", CALCULATE([Total Fraud Transactions])
        ),
        [FraudCnt],
        0.9
    )
RETURN
    DIVIDE(
        CALCULATE(
            [Total Fraud Transactions],
            FILTER(
                SUMMARIZE(fact_transactions, fact_transactions[account_key]),
                CALCULATE([Total Fraud Transactions]) >= Top10Threshold
            )
        ),
        TotalFraud,
        0
    ) * 100
```

---

## 🎯 Model Performance (False Positive Mitigation)

### False Positive Rate
```dax
False Positive Rate =
DIVIDE(
    CALCULATE(
        COUNTROWS(fact_transactions),
        fact_transactions[is_fraud] = 0,
        fact_transactions[fraud_probability] >= 0.5
    ),
    CALCULATE(
        COUNTROWS(fact_transactions),
        fact_transactions[is_fraud] = 0
    ),
    0
) * 100
```

### Precision (Fraud Detection)
```dax
Model Precision =
VAR TP = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[is_fraud]=1, fact_transactions[fraud_probability]>=0.5)
VAR FP = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[is_fraud]=0, fact_transactions[fraud_probability]>=0.5)
RETURN DIVIDE(TP, TP + FP, 0)
```

### Recall (Fraud Detection)
```dax
Model Recall =
VAR TP = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[is_fraud]=1, fact_transactions[fraud_probability]>=0.5)
VAR FN = CALCULATE(COUNTROWS(fact_transactions), fact_transactions[is_fraud]=1, fact_transactions[fraud_probability]<0.5)
RETURN DIVIDE(TP, TP + FN, 0)
```

### F1-Score
```dax
Model F1 Score =
VAR P = [Model Precision]
VAR R = [Model Recall]
RETURN DIVIDE(2 * P * R, P + R, 0)
```

---

## 🌍 Geo Analytics

### Transactions per Location Cell
```dax
Transactions per Location =
AVERAGEX(
    dim_locations,
    dim_locations[transaction_count]
)
```

---

## 💡 Conditional Formatting Helpers

### Risk Color (for conditional formatting)
```dax
Risk Color =
SWITCH(
    SELECTEDVALUE(fact_transactions[risk_tier]),
    "LOW",      "#22C55E",  -- green
    "MED",      "#F59E0B",  -- amber
    "HIGH",     "#EF4444",  -- red
    "CRITICAL", "#7C3AED",  -- purple
    "#94A3B8"               -- grey (unscored)
)
```

### Fraud Flag Label
```dax
Fraud Label =
IF([Total Fraud Transactions] > 0, "⚠️ FRAUD DETECTED", "✅ CLEAN")
```
