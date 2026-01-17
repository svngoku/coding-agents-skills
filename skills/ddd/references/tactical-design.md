# Tactical Design

Tactical design provides building blocks for implementing domain models within a bounded context.

## Table of Contents
1. [Entities](#entities)
2. [Value Objects](#value-objects)
3. [Aggregates](#aggregates)
4. [Domain Services](#domain-services)
5. [Repositories](#repositories)
6. [Domain Events](#domain-events)
7. [Factories](#factories)
8. [Specifications](#specifications)
9. [Decision Guide](#decision-guide)

---

## Entities

Objects defined by identity that persists through state changes.

### Characteristics
- **Identity**: Unique identifier (UUID, natural key)
- **Mutability**: State changes over lifecycle
- **Continuity**: Same entity even if all attributes change

### Implementation Pattern

```python
# Python
from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass
class User:
    id: UUID = field(default_factory=uuid4)
    email: str
    name: str
    status: UserStatus = UserStatus.PENDING
    
    def activate(self) -> None:
        if self.status != UserStatus.PENDING:
            raise InvalidStateError("Can only activate pending users")
        self.status = UserStatus.ACTIVE
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, User):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        return hash(self.id)
```

```typescript
// TypeScript
class User {
  constructor(
    public readonly id: UserId,
    private _email: Email,
    private _name: string,
    private _status: UserStatus = UserStatus.Pending
  ) {}

  activate(): void {
    if (this._status !== UserStatus.Pending) {
      throw new InvalidStateError("Can only activate pending users");
    }
    this._status = UserStatus.Active;
  }

  equals(other: User): boolean {
    return this.id.equals(other.id);
  }
}
```

---

## Value Objects

Objects defined by attributes, with no conceptual identity.

### Characteristics
- **Immutability**: Never changes after creation
- **Equality by value**: Equal if all attributes equal
- **Self-validation**: Always valid, created via factory/constructor
- **Side-effect free**: Methods return new instances

### Implementation Pattern

```python
# Python with Pydantic
from pydantic import BaseModel, field_validator
from typing import Self

class Money(BaseModel, frozen=True):
    amount: Decimal
    currency: str
    
    @field_validator('amount')
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v
    
    def add(self, other: Self) -> Self:
        if self.currency != other.currency:
            raise CurrencyMismatchError()
        return Money(amount=self.amount + other.amount, currency=self.currency)
    
    def multiply(self, factor: Decimal) -> Self:
        return Money(amount=self.amount * factor, currency=self.currency)
```

```typescript
// TypeScript
class Money {
  private constructor(
    public readonly amount: number,
    public readonly currency: Currency
  ) {
    if (amount < 0) throw new InvalidMoneyError("Amount cannot be negative");
  }

  static create(amount: number, currency: Currency): Money {
    return new Money(amount, currency);
  }

  add(other: Money): Money {
    if (!this.currency.equals(other.currency)) {
      throw new CurrencyMismatchError();
    }
    return new Money(this.amount + other.amount, this.currency);
  }

  equals(other: Money): boolean {
    return this.amount === other.amount && this.currency.equals(other.currency);
  }
}
```

### Common Value Objects
- `Money` (amount + currency)
- `Address` (street, city, postal, country)
- `DateRange` (start, end)
- `EmailAddress`, `PhoneNumber`
- `Coordinates` (lat, lng)
- `Quantity`, `Percentage`

---

## Aggregates

Cluster of entities and value objects with defined boundaries and a root entity.

### Aggregate Rules

1. **Single root**: External access only through aggregate root
2. **Consistency boundary**: Invariants enforced within aggregate
3. **Transactional boundary**: One aggregate = one transaction
4. **Reference by ID**: Aggregates reference other aggregates by ID only
5. **Small aggregates**: Prefer smaller, focused aggregates

### Implementation Pattern

```python
# Python
@dataclass
class Order:  # Aggregate Root
    id: OrderId
    customer_id: CustomerId  # Reference by ID, not object
    lines: list[OrderLine] = field(default_factory=list)
    status: OrderStatus = OrderStatus.DRAFT
    _events: list[DomainEvent] = field(default_factory=list, repr=False)
    
    def add_line(self, product_id: ProductId, quantity: int, price: Money) -> None:
        if self.status != OrderStatus.DRAFT:
            raise OrderNotEditableError()
        line = OrderLine(product_id=product_id, quantity=quantity, unit_price=price)
        self.lines.append(line)
    
    def place(self) -> None:
        if not self.lines:
            raise EmptyOrderError()
        self.status = OrderStatus.PLACED
        self._events.append(OrderPlaced(order_id=self.id, placed_at=datetime.utcnow()))
    
    @property
    def total(self) -> Money:
        return sum((line.subtotal for line in self.lines), Money.zero("USD"))
    
    def collect_events(self) -> list[DomainEvent]:
        events = self._events.copy()
        self._events.clear()
        return events

@dataclass(frozen=True)
class OrderLine:  # Entity within aggregate
    product_id: ProductId
    quantity: int
    unit_price: Money
    
    @property
    def subtotal(self) -> Money:
        return self.unit_price.multiply(self.quantity)
```

### Aggregate Design Heuristics

| Question | If Yes | If No |
|----------|--------|-------|
| Must be consistent immediately? | Same aggregate | Separate aggregates + events |
| Changed together frequently? | Same aggregate | Separate aggregates |
| Large number of items? | Separate aggregate | Can be in same |
| Different lifecycles? | Separate aggregates | Can be in same |

---

## Domain Services

Stateless operations that don't naturally belong to any entity.

### When to Use
- Operation involves multiple aggregates
- Operation requires external information
- Significant domain logic that doesn't fit an entity

```python
# Python
class PricingService:
    def __init__(self, discount_repository: DiscountRepository):
        self._discounts = discount_repository
    
    def calculate_order_total(
        self, 
        order: Order, 
        customer: Customer
    ) -> Money:
        base_total = order.total
        discount = self._discounts.find_applicable(customer.tier, order.total)
        return base_total.subtract(discount.apply(base_total))
```

```typescript
// TypeScript
class PricingService {
  constructor(private discountRepository: DiscountRepository) {}

  calculateOrderTotal(order: Order, customer: Customer): Money {
    const baseTotal = order.total;
    const discount = this.discountRepository.findApplicable(
      customer.tier, 
      order.total
    );
    return baseTotal.subtract(discount.apply(baseTotal));
  }
}
```

---

## Repositories

Collection-like interface for aggregate persistence.

### Repository Contract

```python
# Python - Port (interface)
from abc import ABC, abstractmethod

class OrderRepository(ABC):
    @abstractmethod
    def find_by_id(self, order_id: OrderId) -> Order | None: ...
    
    @abstractmethod
    def save(self, order: Order) -> None: ...
    
    @abstractmethod
    def find_by_customer(self, customer_id: CustomerId) -> list[Order]: ...
```

```typescript
// TypeScript - Port (interface)
interface OrderRepository {
  findById(orderId: OrderId): Promise<Order | null>;
  save(order: Order): Promise<void>;
  findByCustomer(customerId: CustomerId): Promise<Order[]>;
}
```

### Repository Rules
- One repository per aggregate root
- Repository interface in domain layer
- Implementation in infrastructure layer
- Returns fully reconstituted aggregates
- Encapsulates query logic

---

## Domain Events

Record of something significant that happened in the domain.

### Event Structure

```python
# Python
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

@dataclass(frozen=True)
class DomainEvent:
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.utcnow)

@dataclass(frozen=True)
class OrderPlaced(DomainEvent):
    order_id: OrderId
    customer_id: CustomerId
    total: Money

@dataclass(frozen=True)
class OrderShipped(DomainEvent):
    order_id: OrderId
    tracking_number: str
    carrier: str
```

```typescript
// TypeScript
abstract class DomainEvent {
  public readonly eventId: string = crypto.randomUUID();
  public readonly occurredAt: Date = new Date();
}

class OrderPlaced extends DomainEvent {
  constructor(
    public readonly orderId: OrderId,
    public readonly customerId: CustomerId,
    public readonly total: Money
  ) {
    super();
  }
}
```

### Event Naming
- Past tense: `OrderPlaced`, `PaymentReceived`, `UserActivated`
- Domain language, not technical: `OrderPlaced` not `OrderCreatedEvent`

---

## Factories

Encapsulate complex creation logic.

```python
# Python
class OrderFactory:
    def __init__(self, pricing_service: PricingService):
        self._pricing = pricing_service
    
    def create_from_cart(self, cart: ShoppingCart, customer: Customer) -> Order:
        order = Order(
            id=OrderId.generate(),
            customer_id=customer.id
        )
        for item in cart.items:
            price = self._pricing.get_price(item.product_id, customer.tier)
            order.add_line(item.product_id, item.quantity, price)
        return order
```

---

## Specifications

Encapsulate business rules for querying or validation.

```python
# Python
from abc import ABC, abstractmethod

class Specification(ABC, Generic[T]):
    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool: ...
    
    def and_(self, other: Specification[T]) -> Specification[T]:
        return AndSpecification(self, other)
    
    def or_(self, other: Specification[T]) -> Specification[T]:
        return OrSpecification(self, other)

class HighValueOrder(Specification[Order]):
    def __init__(self, threshold: Money):
        self._threshold = threshold
    
    def is_satisfied_by(self, order: Order) -> bool:
        return order.total >= self._threshold

# Usage
high_value = HighValueOrder(Money(1000, "USD"))
priority_orders = [o for o in orders if high_value.is_satisfied_by(o)]
```

---

## Decision Guide

### Entity vs Value Object

| Choose Entity When | Choose Value Object When |
|--------------------|--------------------------|
| Identity matters | Only attributes matter |
| Tracked over time | Interchangeable if equal |
| Has lifecycle | Created, used, discarded |
| Referenced specifically | Can be replaced by equal |

### Aggregate Sizing

```
Too Big:
- Concurrency conflicts
- Performance issues  
- Complex transactions

Too Small:
- Invariants spread across aggregates
- Excessive eventual consistency
- Complex event choreography

Right Size:
- Protects true invariants
- Minimal entities
- Clear consistency boundary
```

### Where Does Logic Belong?

| Logic Type | Location |
|------------|----------|
| Single entity invariant | Entity method |
| Value calculation | Value Object |
| Aggregate invariant | Aggregate Root |
| Cross-aggregate operation | Domain Service |
| Orchestration/workflow | Application Service |
| Data transformation | Infrastructure |