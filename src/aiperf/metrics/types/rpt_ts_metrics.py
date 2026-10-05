# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-level metrics for SAP RPT time-series inference."""

from __future__ import annotations

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


# =============================================================================
# Series Count
# =============================================================================


class RPTSeriesCountMetric(BaseRecordMetric[int]):
    """Number of time series contained in one RPT request."""

    flags = MetricFlags.TIME_SERIES_ONLY
    tag = "series_count"
    header = "Series / Request"
    short_header = "Series"
    unit = GenericMetricUnit.COUNT
    display_order = 3000
    console_group = MetricConsoleGroup.DEFAULT

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        series_count = record.request.request_info.series_count

        if series_count is None:
            raise NoMetricValue(
                "RPT series count is not available on RecordContext"
            )

        return series_count


# =============================================================================
# Input Timepoints
# =============================================================================


class RPTInputTimepointsMetric(BaseRecordMetric[int]):
    """Number of historical target observations sent in one RPT request."""

    flags = MetricFlags.TIME_SERIES_ONLY
    tag = "input_timepoints"
    header = "Input Timepoints / Request"
    short_header = "Input TP"
    unit = GenericMetricUnit.COUNT
    display_order = 3010
    console_group = MetricConsoleGroup.DEFAULT

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        input_timepoints = (
            record.request.request_info.input_timepoints
        )

        if input_timepoints is None:
            raise NoMetricValue(
                "RPT input timepoints are not available on RecordContext"
            )

        return input_timepoints


# =============================================================================
# Forecast Timepoints
# =============================================================================


class RPTForecastTimepointsMetric(BaseRecordMetric[int]):
    """Number of forecast target observations requested in one RPT request."""

    flags = MetricFlags.TIME_SERIES_ONLY
    tag = "forecast_timepoints"
    header = "Forecast Timepoints / Request"
    short_header = "Forecast TP"
    unit = GenericMetricUnit.COUNT
    display_order = 3020
    console_group = MetricConsoleGroup.DEFAULT

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        forecast_timepoints = (
            record.request.request_info.forecast_timepoints
        )

        if forecast_timepoints is None:
            raise NoMetricValue(
                "RPT forecast timepoints are not available on RecordContext"
            )

        return forecast_timepoints