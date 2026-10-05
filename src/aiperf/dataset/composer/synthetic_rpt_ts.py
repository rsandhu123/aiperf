# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic dataset composer for SAP RPT time-series requests."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from aiperf.common import random_generator as rng
from aiperf.common.models import Conversation, Turn
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset import SyntheticDataset
from aiperf.config.dataset.content import RPTTimeSeriesConfig
from aiperf.dataset.composer.base import BaseDatasetComposer
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _expected_int(distribution: object) -> int:
    """Return expected value from a sampling distribution or scalar."""
    if isinstance(distribution, (int, float)):
        return int(distribution)

    return int(
        getattr(
            distribution,
            "expected_value",
        )
    )


def _stddev_int(distribution: object) -> int:
    """Return stddev from a sampling distribution or scalar."""
    if isinstance(distribution, (int, float)):
        return 0

    return int(
        getattr(
            distribution,
            "stddev",
            0,
        )
        or 0
    )


class SyntheticRPTTimeSeriesDatasetComposer(
    BaseDatasetComposer
):
    """Generate synthetic SAP RPT /predict requests.

    Each generated request may contain:

        {
            "horizon": <forecast horizon>,
            "data": [
                {
                    "target": [...],
                    "timestamp": [...],
                    "past_covariates": {...},
                    "future_covariates": {...},
                },
                ...
            ],
            "options": {
                "rules": [...],
                "batch_size": <batch size>,
                "output_quantile": <bool>,
                "point_forecast": <string>,
                "quantile_levels": [...],
                "clip_min": <number>,
                "clip_max": <number>,
                "confidence_interval": <number | null>,
                "cross_learning": <bool>,
                "enhanced_forecast": <bool>,
                "hierarchical_labels": [...],
                "hierarchical_levels": [...],
            }
        }

    Target values retain the existing deterministic pattern:

        - deterministic linear upward trend
        - period-31 sawtooth seasonality
        - no random noise

    Covariates are configurable through RPTTimeSeriesConfig.

    A configured past covariate is generated with input_length values.
    A configured future covariate is generated with horizon values.

    All series within a request use the same generated target and
    covariate values.
    """

    def __init__(self, *, run: BenchmarkRun, tokenizer: Tokenizer | None, **kwargs: Any):
        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

        dataset = run.cfg.get_default_dataset()

        if not isinstance(dataset, SyntheticDataset):
            raise ValueError(
                "SyntheticRPTTimeSeriesDatasetComposer requires a synthetic dataset."
            )

        self._config = (
            dataset.rpt_ts
            or RPTTimeSeriesConfig()
        )

        self._num_entries = dataset.entries

        seed = (
            dataset.random_seed
            if dataset.random_seed is not None
            else run.random_seed
        )

        self.session_id_generator = (
            SessionIDGenerator(
                seed=seed,
            )
        )

        # Independent RNG streams for workload dimensions.
        self._series_rng = rng.derive("dataset.rpt_ts.series_count")

        self._length_rng = rng.derive("dataset.rpt_ts.input_length")

        self._horizon_rng = rng.derive("dataset.rpt_ts.horizon")

    def create_dataset(self, ) -> list[Conversation]:
        """Generate synthetic RPT conversations."""

        conversations: list[Conversation] = []

        for _ in range(self._num_entries):
            payload = self._create_payload()
            logger.info(
                json.dumps(
                    payload,
                    indent=2,
                )
            )

            turn = Turn(role="user", raw_payload=payload)

            self._finalize_turn(turn)

            conversation = Conversation(
                session_id=(self.session_id_generator.next()),
                turns=[
                    turn,
                ],
            )

            conversations.append(conversation)

        self._finalize_conversations(conversations)

        return conversations

    def _create_payload(self) -> dict[str, Any]:
        """Create one complete /predict request payload."""

        series_count = self._sample_positive_int(self._config.series_count, self._series_rng)

        input_length = self._sample_positive_int(self._config.input_length, self._length_rng)

        horizon = self._sample_positive_int(self._config.horizon, self._horizon_rng)

        self._validate_dimensions(input_length=input_length, horizon=horizon)

        # Generate once and reuse across the series in this request.
        target = self._generate_target(length=input_length)

        timestamp = self._generate_timestamps(length=input_length)

        past_covariates = self._generate_covariates(config=getattr(self._config, "past_covariates", None), length=input_length)

        future_covariates = self._generate_covariates(config=getattr(self._config, "future_covariates", None), length=horizon)

        self._validate_covariates(past_covariates=past_covariates, future_covariates=future_covariates, input_length=input_length, horizon=horizon)

        data: list[dict[str, Any]] = []

        for _ in range(series_count):
            series: dict[str, Any] = {
                "target": deepcopy(target),
            }

            if timestamp is not None:
                series["timestamp"] = deepcopy(
                    timestamp
                )

            if past_covariates:
                series["past_covariates"] = (
                    deepcopy(
                        past_covariates
                    )
                )

            if future_covariates:
                series["future_covariates"] = (
                    deepcopy(
                        future_covariates
                    )
                )

            data.append(series)

        payload: dict[str, Any] = {
            "horizon": horizon,
            "data": data,
        }

        options = self._create_options(
            series_count=series_count,
        )

        if options:
            payload["options"] = options

        return payload

    def _create_options(
        self,
        *,
        series_count: int,
    ) -> dict[str, Any]:
        """Create the optional RPT request options block."""

        options: dict[str, Any] = {}

        self._add_option(
            options,
            "rules",
            getattr(
                self._config,
                "rules",
                None,
            ),
        )

        self._add_option(
            options,
            "batch_size",
            getattr(
                self._config,
                "batch_size",
                None,
            ),
        )

        self._add_option(
            options,
            "output_quantile",
            getattr(
                self._config,
                "output_quantile",
                None,
            ),
        )

        self._add_option(
            options,
            "point_forecast",
            getattr(
                self._config,
                "point_forecast",
                None,
            ),
        )

        self._add_option(
            options,
            "quantile_levels",
            getattr(
                self._config,
                "quantile_levels",
                None,
            ),
        )

        self._add_option(
            options,
            "clip_min",
            getattr(
                self._config,
                "clip_min",
                None,
            ),
        )

        self._add_option(
            options,
            "clip_max",
            getattr(
                self._config,
                "clip_max",
                None,
            ),
        )

        # confidence_interval is special because None is a meaningful
        # API value: explicitly setting null disables interval output.
        #
        # If your config model needs to distinguish "not configured"
        # from "configured as null", use an explicit sentinel there.
        if hasattr(
            self._config,
            "confidence_interval",
        ):
            confidence_interval = getattr(
                self._config,
                "confidence_interval",
            )

            if confidence_interval is not None:
                options["confidence_interval"] = (
                    confidence_interval
                )

        self._add_option(
            options,
            "cross_learning",
            getattr(
                self._config,
                "cross_learning",
                None,
            ),
        )

        self._add_option(
            options,
            "enhanced_forecast",
            getattr(
                self._config,
                "enhanced_forecast",
                None,
            ),
        )

        hierarchical_labels = getattr(
            self._config,
            "hierarchical_labels",
            None,
        )

        hierarchical_levels = getattr(
            self._config,
            "hierarchical_levels",
            None,
        )

        if hierarchical_labels is not None:
            if (
                len(hierarchical_labels)
                != series_count
            ):
                raise ValueError(
                    "hierarchical_labels must contain "
                    "one entry per generated series: "
                    f"expected {series_count}, got "
                    f"{len(hierarchical_labels)}"
                )

            options["hierarchical_labels"] = (
                deepcopy(
                    hierarchical_labels
                )
            )

        if hierarchical_levels is not None:
            options["hierarchical_levels"] = (
                deepcopy(
                    hierarchical_levels
                )
            )

        self._validate_options(
            options=options,
        )

        return options

    def _generate_timestamps(
        self,
        *,
        length: int,
    ) -> list[int] | None:
        """Generate regular integer timestamps when enabled.

        Integer timestamps provide a simple synthetic regular grid.

        Set ``timestamp: false`` / ``include_timestamp: false``
        in configuration to omit timestamps entirely.
        """

        include_timestamp = getattr(
            self._config,
            "include_timestamp",
            False,
        )

        if not include_timestamp:
            return None

        timestamp_start = getattr(
            self._config,
            "timestamp_start",
            1,
        )

        timestamp_step = getattr(
            self._config,
            "timestamp_step",
            1,
        )

        if timestamp_step <= 0:
            raise ValueError(
                "timestamp_step must be > 0"
            )

        return [
            timestamp_start
            + index * timestamp_step
            for index in range(length)
        ]

    @staticmethod
    def _generate_covariates(
        *,
        config: dict[str, Any] | None,
        length: int,
    ) -> dict[str, list[Any]]:
        """Generate configured synthetic covariates.

        Supported configuration for each covariate:

        Literal list:

            past_covariates:
              temperature:
                [10, 11, 12, 13]

        For synthetic datasets, fixed/scalar configuration:
            past_covariates:
              is_holiday:
                value: 0

        Linear:
            past_covariates:
              temperature:
                pattern: linear
                start: 10
                step: 0.5

        Sawtooth:
            past_covariates:
              demand_signal:
                pattern: sawtooth
                base: 10
                trend: 0.01
                period: 31
                amplitude: 0.0625

        Cyclic:
            past_covariates:
              weekday:
                pattern: cycle
                values: [0, 1, 2, 3, 4, 5, 6]

        Explicit lists must already have the required length.
        Synthetic patterns are expanded to ``length``.
        """

        if not config:
            return {}

        result: dict[str, list[Any]] = {}

        for name, definition in config.items():
            if isinstance(definition, list):
                if len(definition) != length:
                    raise ValueError(f"Covariate '{name}' must have {length} values, got {len(definition)}")

                result[name] = list(definition)
                continue

            if not isinstance(definition, dict):
                # Convenient shorthand:
                #
                #   is_holiday: 0
                #
                # expands to a constant covariate.
                result[name] = [definition for _ in range(length)]
                continue

            pattern = definition.get("pattern", "constant")

            if pattern == "constant":
                value = definition.get("value", 0.0)

                result[name] = [value for _ in range(length)]

            elif pattern == "linear":
                start = float(definition.get("start", 0.0))
                step = float(definition.get("step", 1.0))

                result[name] = [round(start + index * step, 6) for index in range(length)]

            elif pattern == "sawtooth":
                base = float(definition.get("base", 10.0))
                trend = float(definition.get("trend", 0.0))
                period = int(definition.get("period", 31))
                amplitude = float(definition.get("amplitude", 0.0625))
                

                if period <= 0:
                    raise ValueError(f"Covariate '{name}': period must be > 0")

                center = (period - 1) / 2.0

                result[name] = [
                    round(base + index * trend + ((index % period) - center) * amplitude, 6)
                    for index in range(length)
                ]

            elif pattern == "cycle":
                values = definition.get("values")

                if not isinstance(values, list) or not values:
                    raise ValueError(f"Covariate '{name}': 'cycle' requires a non-empty 'values' list"
                )

                result[name] = [
                    values[index % len(values)]
                    for index in range(length)
                ]

            else:
                raise ValueError(f"Unsupported covariate pattern '{pattern}' for '{name}'"
                )

        return result

    @staticmethod
    def _validate_covariates(*, past_covariates: dict[str, list[Any]], future_covariates: dict[str, list[Any]],
        input_length: int,
        horizon: int,
    ) -> None:
        """Validate RPT covariate constraints."""

        for name, values in past_covariates.items():
            if len(values) != input_length:
                raise ValueError(f"past_covariates['{name}'] must contain {input_length} values, got {len(values)}")

        for name, values in future_covariates.items():
            if name not in past_covariates:
                raise ValueError(f"Every future_covariates key must also exist in past_covariates. Missing past covariate: '{name}'")

            if len(values) != horizon:
                raise ValueError(f"future_covariates['{name}'] must contain {horizon} values, got {len(values)}")

    @staticmethod
    def _validate_dimensions(
        *,
        input_length: int,
        horizon: int,
    ) -> None:
        """Validate service dimension constraints."""

        if input_length <= 0:
            raise ValueError("RPT input_length must be > 0")

        if input_length > 8192:
            raise ValueError(f"RPT input_length must be <= 8192, got {input_length}")

        if horizon <= 0:
            raise ValueError("RPT horizon must be > 0")

        if horizon > 1024:
            raise ValueError(f"RPT horizon must be <= 1024, got {horizon}")


    @staticmethod
    def _validate_options(*, options: dict[str, Any],) -> None:
        """Validate configured request options."""

        batch_size = options.get("batch_size")

        if (
            batch_size is not None
            and not (
                1 <= batch_size <= 512
            )
        ):
            raise ValueError("batch_size must be between 1 and 512")

        point_forecast = options.get("point_forecast")
        if point_forecast is not None and point_forecast not in {"median","mean","hierarchical_forecast"}:
            raise ValueError("Point_forecast must be one of: 'median', 'mean', 'hierarchical_forecast'")

        quantile_levels = options.get("quantile_levels")

        if quantile_levels is not None:
            for level in quantile_levels:
                if not 0 < level < 1:
                    raise ValueError("quantile_levels values must be in (0, 1)")

        clip_min = options.get("clip_min")
        if clip_min is not None and not 0 < clip_min < 1:
            raise ValueError("clip_min must be in (0, 1)")

        clip_max = options.get("clip_max")
        if clip_max is not None and not 0 < clip_max < 1:
            raise ValueError("clip_max must be in (0, 1)")

        if (
            clip_min is not None
            and clip_max is not None
            and clip_min > clip_max
        ):
            raise ValueError("clip_min must not be greater than clip_max")

        confidence_interval = options.get("confidence_interval")

        if (
            confidence_interval is not None
            and not (
                0 < confidence_interval < 1
            )
        ):
            raise ValueError("confidence_interval must be in (0, 1)")

        rules = options.get("rules")
        

        if rules is not None:
            valid_operators = {"==", "!=", ">=", "<=", ">", "<"}

            for rule in rules:
                operator = rule.get("operator")

                if operator not in valid_operators:
                    raise ValueError(f"Invalid rule operator '{operator}'")

        if point_forecast == "hierarchical_forecast":
            if not options.get("hierarchical_labels"):
                raise ValueError("hierarchical_labels are required for hierarchical_forecast")

            if not options.get("hierarchical_levels"):
                raise ValueError("hierarchical_levels are required for hierarchical_forecast")

    @staticmethod
    def _add_option(options: dict[str, Any], name: str, value: Any) -> None:
        """Add an option only when explicitly configured."""

        if value is not None:
            options[name] = deepcopy(value)

    @staticmethod
    def _generate_target(
        *,
        length: int,
    ) -> list[float]:
        """Generate deterministic synthetic time-series values.

        Retains the existing benchmark target:
            base =
                10.0 + index * 0.015625

            seasonal =
                ((index % 31) - 15) * 0.0625

            target =
                base + seasonal

        This produces:

            - a linear upward trend
            - a repeating period-31 sawtooth
            - no random noise
        """

        values: list[float] = []

        for index in range(length):
            base = (10.0 + index * 0.015625)

            seasonal = ((index % 31) - 15) * 0.0625

            values.append(round(base + seasonal, 6))

        return values

    @staticmethod
    def _sample_positive_int(
        distribution: object,
        sampler: Any,
    ) -> int:
        """Sample a positive integer or return a configured scalar."""

        if isinstance(distribution, (int, float)):
            value = int(distribution)

            if value <= 0:
                raise ValueError(
                    f"Expected a positive integer, got {value}"
                )

            return value

        mean = _expected_int(distribution)
        stddev = _stddev_int(distribution)

        return sampler.sample_positive_normal_integer(
            mean,
            stddev,
        )