#!/usr/bin/env python3
"""Synthetic public-API checks for clairic==0.1.0; never call live inference."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import date
import json
import os
from typing import Annotated
import unittest
from unittest.mock import patch
import warnings

import httpx
from pydantic import BaseModel, Field, ValidationError

from clairic import (
    APIError,
    APITimeoutError,
    AsyncClairic,
    AuthenticationError,
    Clairic,
    ClairicConfigurationError,
    ClairicTruncationWarning,
    ContextWindowExceededError,
    DateAnswer,
    DatetimeAnswer,
    Duration,
    DurationAnswer,
    Estimate,
    EstimateAnswer,
    Forecast,
    ForecastAnswer,
    InvalidRequestError,
    ModelDefinitionError,
    Number,
    NumberAnswer,
    Result,
    TextAnswer,
    TextListAnswer,
    TimeAnswer,
    __version__,
)


STATE = "Invoice INV-42. Total USD 10. Due 2026-11-25."
DUMMY_AUTH = {"account_id": "offline-account", "api_token": "offline-not-a-real-token"}


class InvoiceExtraction(BaseModel):
    invoice_number: TextAnswer = Field(alias="invoice.number")
    total: Annotated[NumberAnswer[float], Number(unit="USD", decimals=2)]
    due_date: DateAnswer = Field(description="The invoice payment due date")


class InvoiceValues(BaseModel):
    invoice_number: str | None
    total: float | None
    due_date: date | None


class RequiredTotal(BaseModel):
    total: float


class AllFieldTypes(BaseModel):
    label: TextAnswer
    amount: NumberAnswer[int]
    day: DateAnswer
    clock: TimeAnswer
    instant: DatetimeAnswer
    elapsed: Annotated[DurationAnswer[int], Duration(unit="hours")]
    tags: TextListAnswer
    estimate: Annotated[EstimateAnswer[int], Estimate(min=1, max=10, step=1)]
    forecast: Annotated[
        ForecastAnswer, Forecast(min=0, max=100, horizon=2, interval="month")
    ]


def envelope() -> dict:
    """Invented fixture values/usage, not observed service predictions."""
    return {
        "success": True,
        "errors": [],
        "messages": [],
        "result": {
            "model": "clairic",
            "truncated": False,
            "answers": {
                "invoice.number": {
                    "type": "text", "value": "INV-42", "guess": "INV-42",
                    "confidence": 0.99, "p_null": 0.01, "span": [8, 14],
                },
                "total": {
                    "type": "number", "value": 10.0, "guess": 10.0,
                    "confidence": 0.99, "p_null": 0.01,
                },
                "due_date": {
                    "type": "date", "value": "2026-11-25", "guess": "2026-11-25",
                    "confidence": 0.98, "p_null": 0.02,
                },
            },
            "usage": {
                "state_tokens": 20, "image_tokens": 0, "schema_tokens": 40,
                "input_tokens": 60, "output_tokens": 0, "passes": 1,
            },
        },
    }


def mock_response(payload: dict) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda _request: httpx.Response(
            200, headers={"cf-ai-req-id": "offline-fixture"}, json=deepcopy(payload)
        )
    )


class OfflineChecks(unittest.TestCase):
    def test_pinned_version(self) -> None:
        self.assertEqual(__version__, "0.1.0")

    def test_payload_types_aliases_round_trip_and_ownership(self) -> None:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200, headers={"cf-ai-req-id": "offline-fixture"}, json=envelope()
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                result = client.extract(
                    STATE, InvoiceExtraction, reference_date=date(2026, 10, 26)
                )
            self.assertFalse(transport.is_closed)
        self.assertTrue(transport.is_closed)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(
            str(request.url),
            "https://api.cloudflare.com/client/v4/accounts/offline-account/"
            "ai/run/@cf/cloudflare/clairic",
        )
        self.assertEqual(
            request.headers["Authorization"], "Bearer offline-not-a-real-token"
        )
        self.assertEqual(json.loads(request.content), {
            "model": "clairic",
            "state": STATE,
            "reference_date": "2026-10-26",
            "fields": {
                "invoice.number": {"type": "text"},
                "total": {"type": "number", "unit": "USD", "decimals": 2},
                "due_date": {"type": "date", "instructions": "The invoice payment due date"},
            },
        })
        self.assertEqual(result.data.total.value, 10.0)
        self.assertEqual(result.data.due_date.value, date(2026, 11, 25))
        span = result.data.invoice_number.span
        self.assertIsNotNone(span)
        self.assertEqual(STATE[span.start:span.end], "INV-42")
        self.assertEqual(result.request_id, "offline-fixture")
        self.assertEqual(result.values_as(InvoiceValues).invoice_number, "INV-42")
        stored = result.model_dump_json()
        self.assertIn("invoice.number", json.loads(stored)["data"])
        self.assertEqual(Result[InvoiceExtraction].model_validate_json(stored), result)

    def test_missing_value_is_not_replaced_by_guess(self) -> None:
        payload = envelope()
        payload["result"]["answers"]["total"].update(value=None, p_null=0.9)
        with httpx.Client(transport=mock_response(payload)) as transport:
            with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                result = client.extract(STATE, InvoiceExtraction)
        self.assertIsNone(result.data.total.value)
        self.assertEqual(result.data.total.guess, 10.0)
        self.assertIsNone(result.values_as(InvoiceValues).total)
        with self.assertRaises(ValidationError):
            result.values_as(RequiredTotal)

    def test_truncation_has_warning_and_flag(self) -> None:
        payload = envelope()
        payload["result"]["truncated"] = True
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ClairicTruncationWarning)
            with httpx.Client(transport=mock_response(payload)) as transport:
                with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                    result = client.extract(STATE, InvoiceExtraction)
        self.assertTrue(result.truncated)
        self.assertTrue(any(w.category is ClairicTruncationWarning for w in caught))

    def test_status_mapping_and_no_automatic_retry(self) -> None:
        for status, expected in (
            (400, InvalidRequestError), (401, AuthenticationError),
            (403, APIError), (413, ContextWindowExceededError),
            (422, InvalidRequestError), (429, APIError), (500, APIError),
        ):
            with self.subTest(status=status):
                calls = []

                def respond(request: httpx.Request) -> httpx.Response:
                    calls.append(request)
                    return httpx.Response(
                        status,
                        headers={"cf-ai-req-id": "offline-error"},
                        json={"success": False, "errors": [
                            {"code": 1234, "message": "Synthetic failure"}
                        ]},
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
                    with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                        with self.assertRaises(expected) as raised:
                            client.extract(STATE, InvoiceExtraction)
                self.assertIs(type(raised.exception), expected)
                self.assertEqual(raised.exception.status_code, status)
                self.assertEqual(raised.exception.request_id, "offline-error")
                self.assertEqual(len(calls), 1)

    def test_timeout_wrapping(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Synthetic timeout", request=request)

        with httpx.Client(transport=httpx.MockTransport(timeout)) as transport:
            with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                with self.assertRaises(APITimeoutError):
                    client.extract(STATE, InvoiceExtraction)

    def test_invalid_model_fails_before_transport(self) -> None:
        class InvalidExtraction(BaseModel):
            label: str

        calls = []

        def respond(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(500)

        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                with self.assertRaises(ModelDefinitionError):
                    client.extract(STATE, InvalidExtraction)
        self.assertEqual(calls, [])

    def test_all_nine_types_compile_through_public_api(self) -> None:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            return httpx.Response(400, json={
                "success": False,
                "errors": [{"code": 1234, "message": "Synthetic stop after compilation"}],
            })

        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                with self.assertRaises(InvalidRequestError):
                    client.extract("Synthetic all-field compilation.", AllFieldTypes)
        self.assertEqual(len(requests), 1)
        fields = requests[0]["fields"]
        self.assertEqual({v["type"] for v in fields.values()}, {
            "text", "number", "date", "time", "datetime", "duration",
            "list", "estimate", "forecast",
        })
        self.assertEqual(fields["amount"]["integer"], True)
        self.assertEqual(fields["elapsed"]["unit"], "hours")
        self.assertEqual(fields["tags"]["items"], "text")
        self.assertEqual(fields["estimate"]["step"], 1)
        self.assertEqual(fields["forecast"]["horizon"], 2)

    def test_incomplete_usage_is_rejected_by_sdk(self) -> None:
        payload = envelope()
        payload["result"]["usage"] = {"input_tokens": 60, "output_tokens": 0}
        with httpx.Client(transport=mock_response(payload)) as transport:
            with Clairic(**DUMMY_AUTH, http_client=transport) as client:
                with self.assertRaises(APIError):
                    client.extract(STATE, InvoiceExtraction)

    def test_async_parsing_and_transport_ownership(self) -> None:
        async def check() -> None:
            async with httpx.AsyncClient(transport=mock_response(envelope())) as transport:
                async with AsyncClairic(**DUMMY_AUTH, http_client=transport) as client:
                    result = await client.extract(STATE, InvoiceExtraction)
                self.assertFalse(transport.is_closed)
                self.assertEqual(result.data.total.value, 10.0)
                self.assertEqual(result.request_id, "offline-fixture")
            self.assertTrue(transport.is_closed)

        asyncio.run(check())

    def test_missing_configuration_without_reading_real_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ClairicConfigurationError):
                Clairic()


if __name__ == "__main__":
    unittest.main(verbosity=2)
