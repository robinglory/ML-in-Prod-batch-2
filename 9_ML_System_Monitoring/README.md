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

### References
- [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [Monitoring Machine Learning Models in Production](https://towardsdatascience.com/monitoring-machine-learning-models-in-production-how-to-track-data-quality-and-integrity-391435c8a299/)
- [Machine Learning Monitoring: What It Is, and What We Are Missing](https://medium.com/data-science/machine-learning-monitoring-what-it-is-and-what-we-are-missing-e644268023ba)
- [Monitoring unstructured data for LLM and NLP](https://medium.com/data-science/monitoring-unstructured-data-for-llm-and-nlp-efff42704e5b)
- [https://medium.com/data-science/how-to-measure-drift-in-ml-embeddings-ee8adfe1e55e](https://medium.com/data-science/how-to-measure-drift-in-ml-embeddings-ee8adfe1e55e)