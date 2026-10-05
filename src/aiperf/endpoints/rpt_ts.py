# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SAP RPT-TS endpoint integration."""

from __future__ import annotations

from typing import Any

import orjson

from aiperf.common.models import (
    InferenceServerResponse,
    ParsedResponse,
    RequestInfo,
)
from aiperf.endpoints.raw_endpoint import RawEndpoint
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RPTTimeSeriesEndpoint(RawEndpoint):
    """Endpoint for SAP Time Series Foundation Model ``POST /ts/predict``.

    Request bodies normally arrive via ``Turn.raw_payload`` and are sent
    verbatim by AIPerf.

    The endpoint also calculates request-side RPT workload metrics before
    dispatch:

    - series_count
    - input_timepoints
    - forecast_timepoints

    These values are stored on RequestInfo and later carried into
    RecordContext.
    """

    def format_payload(
        self,
        request_info: RequestInfo,
    ) -> dict[str, Any]:
        return super().format_payload(request_info)

    @staticmethod
    def _count_target_timepoints(
        target: Any,
    ) -> int:
        """Count historical target observations for one series.

        Univariate: [1.0, 2.0, 3.0] -> 3
        Multivariate: [[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]] -> 6
        """

        if not isinstance(target, list) or not target:
            return 0

        first = target[0]

        if not isinstance(first, list):
            return len(target)
        return sum(
            len(variate)
            for variate in target
            if isinstance(variate, list)
        )

    @staticmethod
    def _count_target_variates(
        target: Any,
    ) -> int:
        """Count target channels / variates for one series.

        Univariate: [1, 2, 3] -> 1
        Multivariate: [[1, 2, 3], [10, 11, 12]] -> 2
        """

        if not isinstance(target, list) or not target:
            return 0

        first = target[0]
        if not isinstance(first, list):
            return 1

        return sum(
            1
            for variate in target
            if isinstance(variate, list)
        )

    @classmethod
    def _calculate_workload_metrics(
        cls,
        payload: dict[str, Any],
    ) -> tuple[int, int, int]:
        """Calculate RPT request-side workload metrics.

        Returns:
            (
                series_count,
                input_timepoints,
                forecast_timepoints,
            )

        Example:
            {
                "horizon": 100,
                "data": [
                    {"target": [...1024 values...]},
                    ...
                    # 256 series
                ],
            }
        gives:
            series_count        = 256
            input_timepoints    = 256 * 1024 = 262144
            forecast_timepoints = 256 * 100  = 25600

        For multivariate targets, forecast_timepoints is calculated as:
            total_target_variates * horizon
        """

        data = payload.get("data")

        if not isinstance(data, list):
            return 0, 0, 0

        horizon = payload.get("horizon")

        if (
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or horizon <= 0
        ):
            horizon = 0

        series_count = 0
        input_timepoints = 0
        total_target_variates = 0

        for series in data:
            if not isinstance(series, dict):
                continue

            target = series.get("target")
            series_count += 1
            input_timepoints += cls._count_target_timepoints(target)
            total_target_variates += cls._count_target_variates(target)

        forecast_timepoints = (total_target_variates * horizon)

        return (
            series_count,
            input_timepoints,
            forecast_timepoints,
        )

    def populate_request_metrics(
        self,
        request_info: RequestInfo,
        payload: Any,
    ) -> None:
        """Populate RPT request-side workload metrics before dispatch."""
        if isinstance(payload, bytes):
            try:
                payload = orjson.loads(payload)
            except orjson.JSONDecodeError as exc:
                raise ValueError(
                    "Unable to decode RPT request payload"
                ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "RPT request payload must be a JSON object"
            )

        (
            series_count,
            input_timepoints,
            forecast_timepoints,
        ) = self._calculate_workload_metrics(payload)

        request_info.series_count = series_count
        request_info.input_timepoints = input_timepoints
        request_info.forecast_timepoints = forecast_timepoints

    def parse_response(
        self,
        response: InferenceServerResponse,
    ) -> ParsedResponse | None:
        body = response.get_json()

        if not isinstance(body, dict):
            text = response.get_text()

            if not text:
                return None

            return ParsedResponse(
                perf_ns=response.perf_ns,
                data=self.make_text_response_data(text),
            )

        # Successful RPT responses always include these fields.
        # For hierarchical forecasting, result is intentionally []
        # and hierarchical_result carries the forecast, so checking
        # truthiness of result would be incorrect.
        if (
            "result" not in body
            and "hierarchical_result" not in body
        ):
            return None

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=self.make_text_response_data(
                orjson.dumps(body).decode("utf-8")
            ),
        )