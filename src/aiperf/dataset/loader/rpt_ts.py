# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Custom dataset loader for SAP RPT time-series requests."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import (
    BaseRawPayloadLoader,
    LoaderProbeData,
)
from aiperf.dataset.loader.models import RawPayload


# =============================================================================
# RPT request validation
# =============================================================================


def _validate_rpt_payload(
    payload: Any,
) -> dict[str, Any]:
    """Perform lightweight validation for an RPT /ts/predict request."""

    if not isinstance(payload, dict):
        raise ValueError("RPT payload must be a JSON object")

    horizon = payload.get("horizon")
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon <= 0
    ):
        raise ValueError("RPT payload requires a positive integer 'horizon'")
    if horizon > 1024:
        raise ValueError("RPT horizon must be <= 1024")

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("RPT payload requires a non-empty 'data' array")
    
    for index, series in enumerate(data):
        if not isinstance(series, dict):
            raise ValueError(f"RPT data[{index}] must be an object")

        if "target" not in series:
            raise ValueError(f"RPT data[{index}] is missing 'target'")

        target = series["target"]
        if not isinstance(target, list) or not target:
            raise ValueError(f"RPT data[{index}].target must be a non-empty array")

    options = payload.get("options")

    if options is not None:
        if not isinstance(options, dict):
            raise ValueError("RPT 'options' must be an object")

        batch_size = options.get("batch_size")

        if batch_size is not None:
            if (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
                or batch_size < 1
                or batch_size > 512
            ):
                raise ValueError("RPT options.batch_size must be null or between 1 and 512")

    return payload


# =============================================================================
# Dataset loader
# =============================================================================


class RPTTimeSeriesDatasetLoader(BaseRawPayloadLoader):
    """Loader for SAP RPT time-series JSONL datasets.

    Every line represents one complete /ts/predict request.
    Each request becomes:

        one conversation
        -> one turn
        -> Turn.raw_payload
    """

    @classmethod
    def can_load(
        cls,
        data: LoaderProbeData | None = None,
        filename: str | Path | None = None,
    ) -> bool:
        """Detect whether input appears to contain RPT requests."""

        if data is not None:
            if not isinstance(data, dict):
                return False

            horizon = data.get("horizon")
            series_data = data.get("data")

            if (
                not isinstance(horizon, int)
                or isinstance(horizon, bool)
                or horizon <= 0
            ):
                return False

            if (
                not isinstance(series_data, list)
                or not series_data
            ):
                return False

            for series in series_data:
                if not isinstance(series, dict):
                    return False

                target = series.get("target")

                if (
                    not isinstance(target, list)
                    or not target
                ):
                    return False

            return True

        if filename is not None:
            path = Path(filename)

            if not path.is_file():
                return False

            try:
                with path.open("rb") as file:
                    for raw_line in file:
                        line = raw_line.strip()

                        if not line:
                            continue

                        payload = orjson.loads(line)

                        return cls.can_load(
                            data=payload
                        )

            except (OSError, ValueError):
                return False

        return False

    def load_dataset(
        self,
    ) -> dict[str, list[RawPayload]]:
        """Load the configured RPT JSONL file."""

        path = Path(self.filename)

        data: dict[str, list[RawPayload]] = (
            defaultdict(list)
        )

        with path.open("rb") as file:
            for line_number, raw_line in enumerate(
                file,
                start=1,
            ):
                line = raw_line.strip()

                if not line:
                    continue

                try:
                    payload = orjson.loads(line)
                except orjson.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line "
                        f"{line_number} in {path}: {exc}"
                    ) from exc

                try:
                    payload = _validate_rpt_payload(
                        payload
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid RPT payload at line "
                        f"{line_number}: {exc}"
                    ) from exc

                # Each line is one independent request/conversation.
                session_id = (
                    self.session_id_generator.next()
                )

                data[session_id].append(
                    RawPayload(
                        payload=payload
                    )
                )

        if not data:
            raise ValueError(
                f"No valid RPT payloads found in {path}"
            )

        self.info(
            f"Loaded {len(data)} RPT-TS request "
            "payloads from file"
        )

        return dict(data)

    def convert_to_conversations(
        self,
        data: dict[str, list[RawPayload]],
    ) -> list[Conversation]:
        """Convert RPT raw payloads into Conversations."""

        conversations: list[Conversation] = []

        for session_id, payloads in data.items():
            turns = [
                Turn(
                    role="user",
                    raw_payload=raw_payload.payload,
                )
                for raw_payload in payloads
            ]

            conversations.append(
                Conversation(
                    session_id=session_id,
                    turns=turns,
                    context_mode=(
                        ConversationContextMode
                        .MESSAGE_ARRAY_WITH_RESPONSES
                    ),
                )
            )

        return conversations