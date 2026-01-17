# DDD Code Review Checklist

Criteria for reviewing DDD implementations. Use as PR checklist or architecture review guide.

## Table of Contents
1. [Quick Checklist](#quick-checklist)
2. [Strategic Design Review](#strategic-design-review)
3. [Tactical Design Review](#tactical-design-review)
4. [Architecture Review](#architecture-review)
5. [Code Quality Review](#code-quality-review)
6. [Common Issues](#common-issues)

---

## Quick Checklist

### Must Pass (Blockers)
- [ ] Domain layer has zero infrastructure dependencies
- [ ] Aggregates enforce their invariants
- [ ] Value objects are immutable
- [ ] Repository interfaces in domain, implementations in infrastructure
- [ ] No anemic domain models (behavior in entities, not just data)

### Should Pass (Warnings)
- [ ] Ubiquitous language used in code
- [ ] Aggregates are small and focused
- [ ] Domain events for cross-aggregate communication
- [ ] One transaction per aggregate
- [ ] Aggregates reference other aggregates by ID only

### Nice to Have
- [ ] Specification pattern for complex queries
- [ ] Factory pattern for complex creation
- [ ] Domain services are stateless
- [ ] Clear bounded context boundaries

---

## Strategic Design Review

### Bounded Context Assessment

| Check | Pass | Fail |
|-------|------|------|
| Context has clear, documented boundaries | ✓ | ✗ |
| Ubiquitous language defined and used | ✓ | ✗ |
| No model leakage between contexts | ✓ | ✗ |
| Context map documented | ✓ | ✗ |
| Integration patterns explicitly chosen | ✓ | ✗ |

### Questions to Ask

**Boundary clarity:**
- Can you explain where this context starts and ends?
- What other contexts does this interact with?
- Are there terms that mean different things in different contexts?

**Language alignment:**
- Do class names match domain expert terminology?
- Would a domain expert understand this code?
- Are there any "translation" comments explaining technical-to-business mappings?

### Red Flags 🚩
- Same entity class used across multiple contexts
- "Shared" packages between bounded contexts
- Database tables accessed by multiple contexts
- Generic terms like `Entity`, `Item`, `Data` instead of domain terms

---

## Tactical Design Review

### Entity Checklist

```
[ ] Has identity (ID field)
[ ] Equality based on identity, not attributes
[ ] Contains behavior, not just getters/setters
[ ] Validates own invariants
[ ] Encapsulates state changes
```

**Good Example:**
```python
class Order:
    def place(self) -> None:
        if not self.lines:
            raise EmptyOrderError()
        self.status = OrderStatus.PLACED
```

**Bad Example:**
```python
class Order:
    status: str  # No encapsulation
    
# Logic outside entity
def place_order(order):
    if not order.lines:
        raise EmptyOrderError()
    order.status = "placed"  # Direct mutation
```

### Value Object Checklist

```
[ ] Immutable (frozen/readonly)
[ ] Equality by all attributes
[ ] Self-validating on creation
[ ] Side-effect free methods (return new instances)
[ ] No identity field
```

**Good Example:**
```typescript
class Money {
  private constructor(readonly amount: number, readonly currency: string) {}
  
  add(other: Money): Money {
    return new Money(this.amount + other.amount, this.currency);
  }
}
```

**Bad Example:**
```typescript
class Money {
  amount: number;  // Mutable!
  currency: string;
  
  add(other: Money): void {
    this.amount += other.amount;  // Mutates!
  }
}
```

### Aggregate Checklist

```
[ ] Clear root entity
[ ] All access through root
[ ] Enforces consistency boundary
[ ] Small (ideally 1-3 entities)
[ ] References other aggregates by ID only
[ ] One aggregate = one transaction
```

**Sizing Questions:**
- Does this aggregate need to be this large?
- Are all entities truly part of the same consistency boundary?
- Could this be split into smaller aggregates?

### Repository Checklist

```
[ ] Interface in domain layer
[ ] Implementation in infrastructure layer
[ ] One repository per aggregate root
[ ] Returns fully reconstituted aggregates
[ ] No query logic leaking into domain
```

**Good Example:**
```python
# domain/repository/order_repository.py
class OrderRepository(Protocol):
    def find_by_id(self, order_id: OrderId) -> Order | None: ...
    def save(self, order: Order) -> None: ...

# infrastructure/persistence/sql_order_repository.py
class SqlOrderRepository(OrderRepository):
    def find_by_id(self, order_id: OrderId) -> Order | None:
        # SQL implementation
```

### Domain Event Checklist

```
[ ] Named in past tense (OrderPlaced, not PlaceOrder)
[ ] Immutable
[ ] Contains all data needed by handlers
[ ] Uses domain language
[ ] Has timestamp and unique ID
```

---

## Architecture Review

### Layer Dependency Check

```
Allowed dependencies:

Interface → Application → Domain ← Infrastructure
              ↓
           Domain

Not allowed:

Domain → Infrastructure  ✗
Domain → Interface       ✗
Application → Interface  ✗
```

### Dependency Injection Check

```
[ ] Domain services receive interfaces, not implementations
[ ] Application services receive repository interfaces
[ ] Infrastructure provides implementations
[ ] No direct instantiation of infrastructure in domain
```

### CQRS Check (if applicable)

```
[ ] Commands and queries clearly separated
[ ] Command handlers return void or ID only
[ ] Query handlers return DTOs, not domain objects
[ ] Read models optimized for queries
[ ] Write models optimized for behavior
```

### Event Sourcing Check (if applicable)

```
[ ] All state changes through events
[ ] Events are immutable facts
[ ] Aggregate can be rebuilt from events
[ ] Event store is append-only
[ ] Snapshots for performance (if needed)
```

---

## Code Quality Review

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Aggregate | Noun (singular) | `Order`, `Customer` |
| Entity | Noun (singular) | `OrderLine`, `Address` |
| Value Object | Noun (descriptive) | `Money`, `EmailAddress` |
| Domain Service | Verb + Noun | `PricingService`, `InventoryChecker` |
| Repository | Aggregate + Repository | `OrderRepository` |
| Domain Event | Noun + Past Participle | `OrderPlaced`, `PaymentReceived` |
| Command | Verb + Noun | `PlaceOrder`, `CancelOrder` |
| Query | Get + Noun | `GetOrderDetails`, `FindOrdersByCustomer` |

### Test Coverage Expectations

| Layer | Coverage | Focus |
|-------|----------|-------|
| Domain | 90%+ | Business rules, invariants |
| Application | 80%+ | Use case orchestration |
| Infrastructure | 70%+ | Integration, mapping |
| Interface | 60%+ | Request/response handling |

### Documentation Check

```
[ ] Ubiquitous language glossary exists
[ ] Aggregate boundaries documented
[ ] Context map available
[ ] Key domain rules documented in code comments
[ ] Architecture decision records (ADRs) for major choices
```

---

## Common Issues

### Issue: Anemic Domain Model

**Symptom:** Entities have only getters/setters, all logic in services
```python
# BAD
class Order:
    status: str
    lines: list

class OrderService:
    def place_order(self, order):
        if not order.lines:
            raise Error()
        order.status = "placed"
```

**Fix:** Move behavior into entities
```python
# GOOD
class Order:
    def place(self):
        if not self._lines:
            raise EmptyOrderError()
        self._status = OrderStatus.PLACED
```

### Issue: God Aggregate

**Symptom:** Single aggregate with 10+ entities, performance issues

**Fix:** Split into smaller aggregates with eventual consistency

### Issue: Infrastructure in Domain

**Symptom:** Domain classes import database, HTTP, or other infrastructure
```python
# BAD
from sqlalchemy import Column  # Infrastructure leak!

class Order:
    id = Column(Integer)
```

**Fix:** Keep domain pure, use separate ORM models
```python
# GOOD - domain/model/order.py
@dataclass
class Order:
    id: OrderId

# infrastructure/persistence/models.py
class OrderModel(Base):
    id = Column(Integer)
```

### Issue: Missing Ubiquitous Language

**Symptom:** Code uses technical terms, domain experts can't read it
```python
# BAD
class OrderDTO:
    def process_record(self): ...
```

**Fix:** Use domain terms
```python
# GOOD
class Order:
    def place(self): ...
```

### Issue: Aggregate Reference by Object

**Symptom:** Aggregate holds reference to another aggregate object
```python
# BAD
class Order:
    customer: Customer  # Full object!
```

**Fix:** Reference by ID only
```python
# GOOD
class Order:
    customer_id: CustomerId  # ID only!
```

### Issue: Transaction Spanning Aggregates

**Symptom:** Single transaction modifies multiple aggregates
```python
# BAD
def place_order(order, inventory):
    order.place()
    inventory.reserve(order.items)  # Same transaction!
    db.commit()
```

**Fix:** Use domain events for eventual consistency
```python
# GOOD
def place_order(order):
    order.place()  # Emits OrderPlaced event
    db.commit()

# Separate handler
def on_order_placed(event):
    inventory.reserve(event.items)
    db.commit()
```

---

## Review Summary Template

```markdown
## DDD Code Review Summary

**PR:** #123
**Reviewer:** @reviewer
**Date:** 2024-01-15

### Strategic Design
- [ ] Bounded context boundaries respected
- [ ] Ubiquitous language used
- [ ] Context integration patterns appropriate

### Tactical Design
- [ ] Entities have behavior
- [ ] Value objects immutable
- [ ] Aggregates enforce invariants
- [ ] Repositories properly abstracted

### Architecture
- [ ] Layer dependencies correct
- [ ] Domain has no infrastructure dependencies
- [ ] Dependency injection used

### Blockers
- List any blocking issues

### Suggestions
- List non-blocking improvements

### Questions
- List questions for author
```