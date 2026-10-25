# Typed Fields and Results

Contract: unofficial `clairic==0.1.0`, Python 3.10+, Pydantic 2.7+, checked
2026-10-26. The pinned compiler, specs, answer types, and result types establish
SDK behavior; the official service can expose more than this wrapper. See the
sources below and [the source ledger](sources-and-compatibility.md).

## Field Declaration Matrix

Import answer types and specs from `clairic`. Every extraction-model field must
be a supported answer type. Generic answers must be parameterized.

| Declaration | API type | Matching spec and options |
| --- | --- | --- |
| `TextAnswer` | `text` | `Text(unit=..., null_threshold=...)` |
| `NumberAnswer[int]` / `[float]` | `number` | `Number(unit=..., null_threshold=..., decimals=...)` |
| `DateAnswer` | `date` | `Date(null_threshold=...)` |
| `TimeAnswer` | `time` | `Time(null_threshold=...)` |
| `DatetimeAnswer` | `datetime` | `Datetime(null_threshold=...)` |
| `DurationAnswer[int]` / `[float]` | `duration` | `Duration(unit="seconds", null_threshold=..., decimals=...)` |
| `TextListAnswer` | `list` of text | `List(unit=..., decimals=...)` |
| `ListAnswer[int]` / `[float]` / `[date]` | `list` of numbers/dates | `List(unit=..., decimals=...)` |
| `EstimateAnswer[int]` / `[float]` | numeric `estimate` | Required `Estimate(min=..., max=..., step=..., scale=..., unit=...)` |
| `DateEstimateAnswer` | date `estimate` | Required `Estimate(min=date(...), max=date(...), step=...)` |
| `ForecastAnswer` | numeric `forecast` | Required `Forecast(min=..., max=..., horizon=..., interval=..., step=..., scale=..., unit=...)` |

Ellipses above enumerate option positions, not runnable values or universally
required arguments. Specs are frozen, **keyword-only** dataclasses. Ordinary
scalar/list specs are optional; estimate/forecast specs are required. Unset
options are omitted. Default duration unit is `"seconds"`.

```python
from datetime import date
from typing import Annotated

from pydantic import BaseModel, Field
from clairic import (
    DateEstimateAnswer, Duration, DurationAnswer, Estimate, EstimateAnswer,
    Forecast, ForecastAnswer, List, ListAnswer, TextAnswer, TextListAnswer,
)


class DeliveryExtraction(BaseModel):
    order_id: TextAnswer = Field(
        alias="order.id", description="The customer's order identifier"
    )
    delivery_days: Annotated[
        EstimateAnswer[int], Estimate(min=1, max=60, step=1, unit="days")
    ]
    arrival: Annotated[
        DateEstimateAnswer,
        Estimate(min=date(2026, 10, 26), max=date(2026, 12, 31), step="month"),
    ]
    handling_time: Annotated[DurationAnswer[float], Duration(unit="hours")]
    tags: TextListAnswer
    line_amounts: Annotated[ListAnswer[float], List(unit="USD", decimals=2)]
    monthly_revenue: Annotated[
        ForecastAnswer,
        Forecast(min=0, max=100000, horizon=3, interval="month", unit="USD"),
    ]
```

Use bounds, units, and evidence appropriate to the actual domain; these are
synthetic examples, not prediction guarantees. Text extraction copies spans,
not arbitrary generated explanations. Use a generation model for open-ended prose.

## Compilation and Validation

- Field instructions come from `Field(description=...)`; request keys come from
  ordinary `Field(alias=...)`. Python attributes keep their original names.
- Use one matching spec per field. The compiler rejects unsupported types,
  mismatched specs, missing estimate/forecast specs, and empty models.
- Names/aliases must match `^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$`. Keep aliases unique;
  the SDK does not detect two fields colliding on one wire name.
- Separate `validation_alias` is unsupported unless equal to the ordinary alias.
- Do not wrap answers in `Optional`, use bare values, or nest arbitrary models.
  Nullability belongs to scalar `.value`, not the answer object.
- `null_threshold` must be greater than 0 and at most 1; the service default is
  0.5. `decimals` is 0–6. Forecast `horizon` is 1–64.
- Bounds must have `min < max`, with both numeric or both Python dates. Numeric
  bounds must be finite; numeric log scale additionally needs `min > 0`.
- Numeric `step` is positive. An `EstimateAnswer[int]` requires an explicit
  whole-number step, such as `step=1`. Calendar `"month"` / `"year"` steps require
  date bounds. Date estimates snap calendar steps to period starts.
- Scale: `"linear"` / `"log"`. Forecast interval: `"step"`, `"minute"`, `"hour"`,
  `"day"`, `"week"`, `"month"`, `"quarter"`, `"year"`.
- Duration units: `"seconds"`, `"minutes"`, `"hours"`, `"days"`, `"weeks"`,
  `"months"`, `"years"`.

Invalid spec values normally raise `ValueError` when constructing the spec;
invalid extraction declarations raise `ModelDefinitionError` before the POST.
These checks are not an exhaustive runtime type-validation guarantee. Dataclass
type annotations do not independently enforce every Python argument type.

## Answer Semantics

### Scalars

Text, number, date, time, datetime, and duration answers expose `type`, nullable
`value`, optional nullable `guess`, `p_null`, and `confidence`.

`p_null` estimates absence. `confidence` estimates correctness **conditional on
a value existing**. A high-confidence guess does not overrule high absence
probability. The service applies `null_threshold`; the client parses the answer
without recomputing nullness.

Text can include `span: Span | None`, with an end-exclusive `(start, end)`:
`state[span.start:span.end]`. A missing text value may still have a candidate
guess/span. Do not promote it to authoritative evidence automatically.

Dates/times parse into Python `date`, `time`, and `datetime` values. SDK docstrings
describe time/datetime without timezone, but the SDK does not enforce a general
timezone-rejection validator. Define timezone policy explicitly. Duration also
has a required `unit`; preserve it with its numeric value.

### Lists

`TextListAnswer` has `value: list[str]`, `spans: list[Span]`, and overall
`confidence`. `ListAnswer` has a list of typed values, per-item `confidences`,
and overall `confidence`. Numeric/date lists do not have source spans.
No match yields an empty list, not a nullable scalar answer.

### Estimates

Numeric estimates expose nullable `value`, float `mean`, typed `median`, `p10`,
`p90`, `p_below`, `p_above`, and optional `quantiles`. Point `value` is the mean
rounded to the step grid, **not necessarily the median**. Tail probabilities
describe mass outside the declared bounds.

Date estimates have date-valued `value`, `mean`, `median`, `p10`, `p90`, and
optional quantiles. They do not declare numeric tail-probability attributes.

### Forecasts

`ForecastAnswer` has `value`, `median`, `p10`, and `p90` lists, plus `steps`.
Each `ForecastStep` contains numeric `value`, `mean`, `median`, `p10`, `p90`,
tail probabilities, and optional quantiles.

Estimates and forecasts do **not** have scalar `.confidence` / `.p_null`.
Their p10–p90 interval is a central 80% model interval, not guaranteed coverage.
Use relevant history/comparables, evaluate calibration and forecast error, and
do not replace a trained time-series model by assumption.

If returned, 99 quantiles represent percentiles 1–99 (indices 9/49/89 correspond
to p10/median/p90). The SDK can parse them but cannot request the official
`include_quantiles` option.

## Result, Export, and Storage

`Result[Extraction]` has `data`, `usage`, `truncated`, `model`, and optional
`request_id` from response header `cf-ai-req-id`.

SDK `Usage` requires six integer fields: `state_tokens`, `image_tokens`,
`schema_tokens`, `input_tokens`, `output_tokens`, and `passes`. The published
raw output schema requires fewer; a schema-valid raw response can therefore
still fail SDK parsing. Do not make incomplete mock usage objects.

`result.values_as(Target)` copies answer `.value` fields with matching **Python
attribute names**, ignores unmatched source fields, and performs target Pydantic
validation. It is not alias-to-alias remapping. Nonnullable targets can fail
for absent values; uncertainty is discarded by this export.

Store the full SDK object with `model_dump_json()` and restore with
`Result[Extraction].model_validate_json(...)`. Serialization uses extraction
wire aliases. Do not feed a raw REST envelope directly into this operation.

Answer/usage types preserve extra data in `model_extra`. Frozen answer/result
objects are not deeply immutable: contained lists and a caller-defined
extraction model may still be mutable.

## Sources

- [Pinned compiler](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/_compile.py)
- [Specs](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/specs.py)
- [Answer types](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/answers.py)
- [Result types](https://github.com/VianneyMI/clairic/blob/d4b0f3390a05e2310f0fc033803e65658abfcfae/src/clairic/result.py)
