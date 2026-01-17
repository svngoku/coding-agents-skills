# Python DDD Patterns

Implementations using popular Python frameworks: Pydantic, SQLAlchemy, FastAPI.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Value Objects with Pydantic](#value-objects-with-pydantic)
3. [Entities and Aggregates](#entities-and-aggregates)
4. [Domain Events](#domain-events)
5. [Repositories with SQLAlchemy](#repositories-with-sqlalchemy)
6. [Application Services](#application-services)
7. [FastAPI Integration](#fastapi-integration)
8. [Dependency Injection](#dependency-injection)
9. [Testing Patterns](#testing-patterns)

---

## Project Setup

### Recommended Structure

```
src/
├── domain/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── order.py           # Aggregate
│   │   ├── order_line.py      # Entity
│   │   └── value_objects.py   # Money, Address, etc.
│   ├── event/
│   │   ├── __init__.py
│   │   └── order_events.py
│   ├── repository/
│   │   ├── __init__.py
│   │   └── order_repository.py  # Interface (Protocol)
│   └── service/
│       ├── __init__.py
│       └── pricing_service.py
├── application/
│   ├── __init__.py
│   ├── command/
│   │   └── place_order.py
│   ├── query/
│   │   └── get_order.py
│   └── dto/
│       └── order_dto.py
├── infrastructure/
│   ├── __init__.py
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── order_repository.py  # Implementation
│   │   └── unit_of_work.py
│   └── messaging/
│       └── event_bus.py
└── interface/
    ├── __init__.py
    └── api/
        ├── __init__.py
        ├── main.py
        └── routes/
            └── orders.py
```

### Dependencies (pyproject.toml)

```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "sqlalchemy>=2.0",
    "fastapi>=0.100",
    "uvicorn>=0.23",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "httpx>=0.24",  # For FastAPI testing
]
```

---

## Value Objects with Pydantic

```python
# domain/model/value_objects.py
from decimal import Decimal
from typing import Self
from pydantic import BaseModel, field_validator, model_validator

class Money(BaseModel, frozen=True):
    """Immutable value object for monetary amounts."""
    amount: Decimal
    currency: str = "USD"
    
    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v.quantize(Decimal("0.01"))
    
    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        if len(v) != 3:
            raise ValueError("Currency must be 3-letter ISO code")
        return v.upper()
    
    def add(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}")
        return Money(amount=self.amount + other.amount, currency=self.currency)
    
    def multiply(self, factor: int | Decimal) -> Self:
        return Money(amount=self.amount * Decimal(factor), currency=self.currency)
    
    @classmethod
    def zero(cls, currency: str = "USD") -> Self:
        return cls(amount=Decimal("0"), currency=currency)


class Address(BaseModel, frozen=True):
    """Immutable address value object."""
    street: str
    city: str
    state: str
    postal_code: str
    country: str = "US"
    
    @model_validator(mode="after")
    def validate_address(self) -> Self:
        if self.country == "US" and len(self.postal_code) not in (5, 10):
            raise ValueError("US postal code must be 5 or 9 digits")
        return self
    
    def format_single_line(self) -> str:
        return f"{self.street}, {self.city}, {self.state} {self.postal_code}"


class OrderId(BaseModel, frozen=True):
    """Typed ID to prevent mixing different entity IDs."""
    value: str
    
    @classmethod
    def generate(cls) -> Self:
        import uuid
        return cls(value=str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value
    
    def __hash__(self) -> int:
        return hash(self.value)


class CustomerId(BaseModel, frozen=True):
    value: str
    
    def __str__(self) -> str:
        return self.value
    
    def __hash__(self) -> int:
        return hash(self.value)
```

---

## Entities and Aggregates

```python
# domain/model/order.py
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from domain.model.value_objects import OrderId, CustomerId, Money
from domain.event.order_events import OrderPlaced, OrderCancelled

if TYPE_CHECKING:
    from domain.event import DomainEvent


class OrderStatus(Enum):
    DRAFT = "draft"
    PLACED = "placed"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class OrderLine:
    """Entity within Order aggregate."""
    product_id: str
    quantity: int
    unit_price: Money
    
    @property
    def subtotal(self) -> Money:
        return self.unit_price.multiply(self.quantity)
    
    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")


@dataclass
class Order:
    """Aggregate root for orders."""
    id: OrderId
    customer_id: CustomerId
    status: OrderStatus = OrderStatus.DRAFT
    lines: list[OrderLine] = field(default_factory=list)
    placed_at: datetime | None = None
    _events: list["DomainEvent"] = field(default_factory=list, repr=False)
    
    # Factory method
    @classmethod
    def create(cls, customer_id: CustomerId) -> "Order":
        return cls(
            id=OrderId.generate(),
            customer_id=customer_id,
        )
    
    # Commands
    def add_line(self, product_id: str, quantity: int, unit_price: Money) -> None:
        if self.status != OrderStatus.DRAFT:
            raise InvalidOrderStateError("Cannot modify non-draft order")
        
        # Check if product already exists
        for line in self.lines:
            if line.product_id == product_id:
                raise DuplicateProductError(f"Product {product_id} already in order")
        
        self.lines.append(OrderLine(
            product_id=product_id,
            quantity=quantity,
            unit_price=unit_price,
        ))
    
    def remove_line(self, product_id: str) -> None:
        if self.status != OrderStatus.DRAFT:
            raise InvalidOrderStateError("Cannot modify non-draft order")
        
        self.lines = [l for l in self.lines if l.product_id != product_id]
    
    def place(self) -> None:
        if self.status != OrderStatus.DRAFT:
            raise InvalidOrderStateError("Can only place draft orders")
        if not self.lines:
            raise EmptyOrderError("Cannot place order without items")
        
        self.status = OrderStatus.PLACED
        self.placed_at = datetime.utcnow()
        self._events.append(OrderPlaced(
            order_id=self.id,
            customer_id=self.customer_id,
            total=self.total,
            placed_at=self.placed_at,
        ))
    
    def cancel(self, reason: str) -> None:
        if self.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            raise InvalidOrderStateError("Cannot cancel shipped/delivered order")
        
        self.status = OrderStatus.CANCELLED
        self._events.append(OrderCancelled(
            order_id=self.id,
            reason=reason,
        ))
    
    # Queries
    @property
    def total(self) -> Money:
        if not self.lines:
            return Money.zero()
        return sum((line.subtotal for line in self.lines[1:]), self.lines[0].subtotal)
    
    @property
    def item_count(self) -> int:
        return sum(line.quantity for line in self.lines)
    
    # Event handling
    def collect_events(self) -> list["DomainEvent"]:
        events = self._events.copy()
        self._events.clear()
        return events


# Exceptions
class OrderError(Exception):
    pass

class InvalidOrderStateError(OrderError):
    pass

class EmptyOrderError(OrderError):
    pass

class DuplicateProductError(OrderError):
    pass
```

---

## Domain Events

```python
# domain/event/__init__.py
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events."""
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)


# domain/event/order_events.py
from dataclasses import dataclass
from datetime import datetime

from domain.event import DomainEvent
from domain.model.value_objects import OrderId, CustomerId, Money


@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    order_id: OrderId
    customer_id: CustomerId
    total: Money
    placed_at: datetime


@dataclass(frozen=True)
class OrderCancelled(DomainEvent):
    order_id: OrderId
    reason: str


@dataclass(frozen=True)
class OrderShipped(DomainEvent):
    order_id: OrderId
    tracking_number: str
    carrier: str
```

---

## Repositories with SQLAlchemy

```python
# domain/repository/order_repository.py
from typing import Protocol

from domain.model.order import Order
from domain.model.value_objects import OrderId, CustomerId


class OrderRepository(Protocol):
    """Port - Repository interface defined in domain layer."""
    
    def find_by_id(self, order_id: OrderId) -> Order | None: ...
    
    def save(self, order: Order) -> None: ...
    
    def find_by_customer(self, customer_id: CustomerId) -> list[Order]: ...


# infrastructure/persistence/models.py
from datetime import datetime
from decimal import Decimal
from sqlalchemy import String, Integer, DateTime, Numeric, ForeignKey, Enum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class OrderModel(Base):
    __tablename__ = "orders"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    placed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    lines: Mapped[list["OrderLineModel"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
    )


class OrderLineModel(Base):
    __tablename__ = "order_lines"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), nullable=False)
    product_id: Mapped[str] = mapped_column(String(36), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    unit_price_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    
    order: Mapped["OrderModel"] = relationship(back_populates="lines")


# infrastructure/persistence/order_repository.py
from sqlalchemy.orm import Session

from domain.model.order import Order, OrderLine, OrderStatus
from domain.model.value_objects import OrderId, CustomerId, Money
from domain.repository.order_repository import OrderRepository
from infrastructure.persistence.models import OrderModel, OrderLineModel


class SqlAlchemyOrderRepository(OrderRepository):
    """Adapter - Repository implementation."""
    
    def __init__(self, session: Session):
        self._session = session
    
    def find_by_id(self, order_id: OrderId) -> Order | None:
        model = self._session.get(OrderModel, str(order_id))
        return self._to_domain(model) if model else None
    
    def save(self, order: Order) -> None:
        model = self._to_model(order)
        self._session.merge(model)
    
    def find_by_customer(self, customer_id: CustomerId) -> list[Order]:
        models = (
            self._session.query(OrderModel)
            .filter(OrderModel.customer_id == str(customer_id))
            .all()
        )
        return [self._to_domain(m) for m in models]
    
    def _to_domain(self, model: OrderModel) -> Order:
        return Order(
            id=OrderId(value=model.id),
            customer_id=CustomerId(value=model.customer_id),
            status=OrderStatus(model.status),
            placed_at=model.placed_at,
            lines=[
                OrderLine(
                    product_id=line.product_id,
                    quantity=line.quantity,
                    unit_price=Money(
                        amount=line.unit_price_amount,
                        currency=line.unit_price_currency,
                    ),
                )
                for line in model.lines
            ],
        )
    
    def _to_model(self, order: Order) -> OrderModel:
        return OrderModel(
            id=str(order.id),
            customer_id=str(order.customer_id),
            status=order.status.value,
            placed_at=order.placed_at,
            lines=[
                OrderLineModel(
                    product_id=line.product_id,
                    quantity=line.quantity,
                    unit_price_amount=line.unit_price.amount,
                    unit_price_currency=line.unit_price.currency,
                )
                for line in order.lines
            ],
        )
```

---

## Application Services

```python
# application/command/place_order.py
from dataclasses import dataclass
from decimal import Decimal

from domain.model.order import Order
from domain.model.value_objects import CustomerId, Money
from domain.repository.order_repository import OrderRepository
from infrastructure.messaging.event_bus import EventBus


@dataclass(frozen=True)
class OrderItemInput:
    product_id: str
    quantity: int
    unit_price: Decimal
    currency: str = "USD"


@dataclass(frozen=True)
class PlaceOrderCommand:
    customer_id: str
    items: list[OrderItemInput]


class PlaceOrderHandler:
    def __init__(
        self,
        order_repository: OrderRepository,
        event_bus: EventBus,
    ):
        self._orders = order_repository
        self._events = event_bus
    
    def handle(self, command: PlaceOrderCommand) -> str:
        order = Order.create(CustomerId(value=command.customer_id))
        
        for item in command.items:
            order.add_line(
                product_id=item.product_id,
                quantity=item.quantity,
                unit_price=Money(amount=item.unit_price, currency=item.currency),
            )
        
        order.place()
        self._orders.save(order)
        
        # Publish events after save
        for event in order.collect_events():
            self._events.publish(event)
        
        return str(order.id)
```

---

## FastAPI Integration

```python
# interface/api/routes/orders.py
from decimal import Decimal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from application.command.place_order import PlaceOrderCommand, PlaceOrderHandler, OrderItemInput
from domain.model.order import OrderError


router = APIRouter(prefix="/orders", tags=["orders"])


class OrderItemRequest(BaseModel):
    product_id: str
    quantity: int
    unit_price: Decimal


class CreateOrderRequest(BaseModel):
    customer_id: str
    items: list[OrderItemRequest]


class CreateOrderResponse(BaseModel):
    order_id: str


@router.post("", response_model=CreateOrderResponse)
def create_order(
    request: CreateOrderRequest,
    handler: PlaceOrderHandler = Depends(),
) -> CreateOrderResponse:
    try:
        command = PlaceOrderCommand(
            customer_id=request.customer_id,
            items=[
                OrderItemInput(
                    product_id=item.product_id,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                )
                for item in request.items
            ],
        )
        order_id = handler.handle(command)
        return CreateOrderResponse(order_id=order_id)
    except OrderError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

---

## Dependency Injection

```python
# infrastructure/config/dependencies.py
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from domain.repository.order_repository import OrderRepository
from infrastructure.persistence.order_repository import SqlAlchemyOrderRepository
from infrastructure.messaging.event_bus import InMemoryEventBus
from application.command.place_order import PlaceOrderHandler


@lru_cache
def get_engine():
    return create_engine("postgresql://user:pass@localhost/db")


def get_session() -> Session:
    SessionLocal = sessionmaker(bind=get_engine())
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_order_repository(session: Session = Depends(get_session)) -> OrderRepository:
    return SqlAlchemyOrderRepository(session)


def get_event_bus() -> InMemoryEventBus:
    return InMemoryEventBus()


def get_place_order_handler(
    order_repo: OrderRepository = Depends(get_order_repository),
    event_bus: InMemoryEventBus = Depends(get_event_bus),
) -> PlaceOrderHandler:
    return PlaceOrderHandler(order_repo, event_bus)
```

---

## Testing Patterns

```python
# tests/domain/test_order.py
import pytest
from decimal import Decimal

from domain.model.order import Order, InvalidOrderStateError, EmptyOrderError
from domain.model.value_objects import CustomerId, Money
from domain.event.order_events import OrderPlaced


class TestOrder:
    def test_create_order(self):
        order = Order.create(CustomerId(value="cust-123"))
        assert order.status.value == "draft"
        assert len(order.lines) == 0
    
    def test_add_line(self):
        order = Order.create(CustomerId(value="cust-123"))
        order.add_line("prod-1", 2, Money(amount=Decimal("10.00")))
        
        assert len(order.lines) == 1
        assert order.total == Money(amount=Decimal("20.00"))
    
    def test_place_order_emits_event(self):
        order = Order.create(CustomerId(value="cust-123"))
        order.add_line("prod-1", 1, Money(amount=Decimal("10.00")))
        
        order.place()
        
        events = order.collect_events()
        assert len(events) == 1
        assert isinstance(events[0], OrderPlaced)
        assert events[0].order_id == order.id
    
    def test_cannot_place_empty_order(self):
        order = Order.create(CustomerId(value="cust-123"))
        
        with pytest.raises(EmptyOrderError):
            order.place()
    
    def test_cannot_modify_placed_order(self):
        order = Order.create(CustomerId(value="cust-123"))
        order.add_line("prod-1", 1, Money(amount=Decimal("10.00")))
        order.place()
        
        with pytest.raises(InvalidOrderStateError):
            order.add_line("prod-2", 1, Money(amount=Decimal("5.00")))


# tests/infrastructure/test_order_repository.py
class TestSqlAlchemyOrderRepository:
    def test_save_and_retrieve(self, session):
        repo = SqlAlchemyOrderRepository(session)
        order = Order.create(CustomerId(value="cust-123"))
        order.add_line("prod-1", 2, Money(amount=Decimal("10.00")))
        
        repo.save(order)
        session.commit()
        
        retrieved = repo.find_by_id(order.id)
        assert retrieved is not None
        assert retrieved.id == order.id
        assert len(retrieved.lines) == 1
```