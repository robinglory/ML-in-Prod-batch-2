from evidently.report import Report
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import TargetDriftPreset
from typing import Text
from evidently.metric_preset import DataDriftPreset


from evidently.metrics import (
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionPredictedVsActualPlot,
    RegressionErrorPlot,
    RegressionAbsPercentageErrorPlot,
    RegressionErrorDistribution,
    RegressionErrorNormality,
    RegressionTopErrorMetric
)


def build_model_performance_taxi_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping
) -> Text:
    

    model_performance_report = Report(metrics=[
        RegressionQualityMetric(),
        RegressionPredictedVsActualScatter(),
        RegressionPredictedVsActualPlot(),
        RegressionErrorPlot(),
        RegressionAbsPercentageErrorPlot(),
        RegressionErrorDistribution(),
        RegressionErrorNormality(),
        RegressionTopErrorMetric()
    ])
    model_performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    report_path = f'/fastapi/reports/model_performance.html'
    model_performance_report.save_html(report_path)

    print("Hello testing")
  
    return report_path



def build_target_drift_taxi_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping
) -> Text:

    target_drift_report = Report(metrics=[TargetDriftPreset(),DataDriftPreset()])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )

   
    report_path = '/fastapi/reports/target_drift.html'
    target_drift_report.save_html(report_path)

    return report_path
