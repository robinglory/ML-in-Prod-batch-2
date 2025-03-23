# Concept Drift vs Data Drift

| Type           | Description | Causes | Examples |
|---------------|------------|--------|----------|
| **Concept Drift** | The relationship between input features and target variable changes over time. | Changes in user behavior, seasonality, economic shifts. | A spam classifier becomes less effective as spammers change tactics. |
| **Data Drift** | The distribution of input features or target variable shifts over time. | Changes in data collection, external factors, pipeline issues. | Sensor readings change due to device degradation. |
| **Feature Drift** | A shift in the distribution of input features without changing their relationship with the target variable. | Changes in user demographics, data preprocessing issues. | A recommendation system sees a shift in product preferences due to trends. |
| **Label Drift** | A shift in the distribution of the target variable while input features remain stable. | External events, market shifts, social trends. | The percentage of loan defaults increases due to an economic downturn. |

## Summary
- **Concept Drift** affects the underlying relationship between features and the target.
- **Data Drift** affects the input features or target distribution without necessarily changing their relationship.
- **Feature Drift** is about shifts in input data.
- **Label Drift** is about shifts in target variable distribution.

### Monitoring & Mitigation Strategies
- Use statistical tests (Kolmogorov-Smirnov, PSI) to detect drift.
- Implement model retraining pipelines for drift adaptation.
- Use adaptive learning techniques for evolving patterns.
