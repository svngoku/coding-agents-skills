# Architecture Patterns

Architectural patterns for organizing DDD implementations. Patterns can be combined based on requirements.

## Table of Contents
1. [Layered Architecture](#layered-architecture)
2. [Hexagonal Architecture](#hexagonal-architecture)
3. [Clean Architecture](#clean-architecture)
4. [CQRS](#cqrs)
5. [Event Sourcing](#event-sourcing)
6. [Pattern Combinations](#pattern-combinations)
7. [Selection Guide](#selection-guide)

---

## Layered Architecture

Traditional approach with horizontal layers.

```
┌─────────────────────────────────────┐
│         Presentation Layer          │  ← UI, API Controllers
├─────────────────────────────────────┤
│         Application Layer           │  ← Use Cases, Orchestration
├─────────────────────────────────────┤
│           Domain Layer              │  ← Business Logic, Entities
├─────────────────────────────────────┤
│        Infrastructure Layer         │  ← Database, External Services
└─────────────────────────────────────┘
         Dependencies flow DOWN
```

### Limitations
- Domain depends on infrastructure
- Hard to test domain in isolation
- Technology changes ripple through layers

---

## Hexagonal Architecture

Also known as Ports & Adapters. Domain at center, isolated from external concerns.

```
                    ┌─────────────┐
                    │   REST API  │
                    │  (Adapter)  │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 ▼                 │
         │         ┌─────────────┐           │
         │         │    Port     │           │
┌────────┴───┐     │  (Input)    │     ┌─────┴────────┐
│    CLI     │────►├─────────────┤◄────│   Message    │
│  (Adapter) │     │             │     │   Queue      │
└────────────┘     │   DOMAIN    │     │  (Adapter)   │
                   │   (Core)    │     └──────────────┘
┌────────────┐     │             │     ┌──────────────┐
│  Database  │◄────┤─────────────┤────►│  External    │
│  (Adapter) │     │    Port     │     │    API       │
└────────────┘     │  (Output)   │     │  (Adapter)   │
                   └─────────────┘     └──────────────┘
         │                                   │
         └───────────────────────────────────┘
```

### Key Concepts

**Ports**: Interfaces defined by the domain
- **Inbound/Driving**: How outside world interacts with domain (use cases)
- **Outbound/Driven**: How domain interacts with infrastructure (repositories)

**Adapters**: Implementations of ports
- **Primary/Driving**: REST controllers, CLI, message handlers
- **Secondary/Driven**: Database repositories, API clients, message publishers

### Project Structure

```
src/
├── domain/                      # Core (no external dependencies)
│   ├── model/
│   │   ├── order.py
│   │   └── customer.py
│   ├── port/
│   │   ├── inbound/
│   │   │   └── order_service.py      # Input port (interface)
│   │   └── outbound/
│   │       ├── order_repository.py   # Output port (interface)
│   │       └── payment_gateway.py    # Output port (interface)
│   └── service/
│       └── order_domain_service.py
│
├── application/                 # Use case implementations
│   └── order_application_service.py  # Implements inbound port
│
└── infrastructure/              # Adapters
    ├── adapter/
    │   ├── inbound/
    │   │   ├── rest/
    │   │   │   └── order_controller.py
    │   │   └── cli/
    │   │       └── order_commands.py
    │   └── outbound/
    │       ├── persistence/
    │       │   └── sql_order_repository.py
    │       └── external/
    │           └── stripe_payment_gateway.py
    └── config/
        └── dependency_injection.py
```

### Implementation Example

```python
# domain/port/inbound/order_service.py (Input Port)
from abc import ABC, abstractmethod

class OrderService(ABC):
    @abstractmethod
    def place_order(self, command: PlaceOrderCommand) -> OrderId: ...

# domain/port/outbound/order_repository.py (Output Port)
class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: Order) -> None: ...
    @abstractmethod
    def find_by_id(self, order_id: OrderId) -> Order | None: ...

# application/order_application_service.py (Implements Input Port)
class OrderApplicationService(OrderService):
    def __init__(
        self,
        order_repo: OrderRepository,
        payment_gateway: PaymentGateway
    ):
        self._orders = order_repo
        self._payments = payment_gateway
    
    def place_order(self, command: PlaceOrderCommand) -> OrderId:
        order = Order.create(command.customer_id, command.items)
        self._payments.authorize(command.payment_details, order.total)
        order.place()
        self._orders.save(order)
        return order.id

# infrastructure/adapter/inbound/rest/order_controller.py (Driving Adapter)
@router.post("/orders")
def create_order(request: CreateOrderRequest, service: OrderService = Depends()):
    command = PlaceOrderCommand.from_request(request)
    order_id = service.place_order(command)
    return {"order_id": str(order_id)}

# infrastructure/adapter/outbound/persistence/sql_order_repository.py (Driven Adapter)
class SqlOrderRepository(OrderRepository):
    def __init__(self, session: Session):
        self._session = session
    
    def save(self, order: Order) -> None:
        record = self._to_record(order)
        self._session.merge(record)
    
    def find_by_id(self, order_id: OrderId) -> Order | None:
        record = self._session.get(OrderRecord, str(order_id))
        return self._to_domain(record) if record else None
```

---

## Clean Architecture

Uncle Bob's architecture with explicit dependency rule.

```
┌─────────────────────────────────────────────────────────────┐
│                    Frameworks & Drivers                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               Interface Adapters                     │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │            Application Business Rules        │    │    │
│  │  │  ┌─────────────────────────────────────┐    │    │    │
│  │  │  │    Enterprise Business Rules        │    │    │    │
│  │  │  │         (Entities)                  │    │    │    │
│  │  │  └─────────────────────────────────────┘    │    │    │
│  │  │            (Use Cases)                      │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │      (Controllers, Gateways, Presenters)            │    │
│  └─────────────────────────────────────────────────────┘    │
│          (Web, DB, Devices, External Interfaces)            │
└─────────────────────────────────────────────────────────────┘
              Dependencies point INWARD only
```

### Layers

| Layer | Contains | Depends On |
|-------|----------|------------|
| **Entities** | Domain models, business rules | Nothing |
| **Use Cases** | Application-specific logic | Entities |
| **Interface Adapters** | Controllers, presenters, gateways | Use Cases, Entities |
| **Frameworks** | Web framework, DB, UI | Everything |

### Use Case Pattern

```python
# use_cases/place_order.py
@dataclass
class PlaceOrderInput:
    customer_id: str
    items: list[OrderItemInput]
    payment_method: str

@dataclass
class PlaceOrderOutput:
    order_id: str
    total: Decimal
    status: str

class PlaceOrderUseCase:
    def __init__(
        self,
        order_repo: OrderRepository,
        customer_repo: CustomerRepository,
        payment_service: PaymentService,
        presenter: PlaceOrderPresenter
    ):
        self._orders = order_repo
        self._customers = customer_repo
        self._payments = payment_service
        self._presenter = presenter
    
    def execute(self, input_data: PlaceOrderInput) -> None:
        customer = self._customers.find_by_id(input_data.customer_id)
        if not customer:
            self._presenter.present_error("Customer not found")
            return
        
        order = Order.create(customer.id)
        for item in input_data.items:
            order.add_line(item.product_id, item.quantity, item.price)
        
        payment_result = self._payments.process(
            input_data.payment_method,
            order.total
        )
        
        if not payment_result.success:
            self._presenter.present_error(payment_result.error)
            return
        
        order.place()
        self._orders.save(order)
        
        output = PlaceOrderOutput(
            order_id=str(order.id),
            total=order.total.amount,
            status=order.status.value
        )
        self._presenter.present_success(output)
```

---

## CQRS

Command Query Responsibility Segregation: separate models for reads and writes.

```
                    ┌─────────────────┐
                    │     Client      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │    Command     │           │     Query      │
     │    (Write)     │           │     (Read)     │
     └────────┬───────┘           └────────┬───────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │    Command     │           │     Query      │
     │    Handler     │           │    Handler     │
     └────────┬───────┘           └────────┬───────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │  Write Model   │           │   Read Model   │
     │  (Aggregates)  │           │    (Views)     │
     └────────┬───────┘           └────────┬───────┘
              │                             │
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │  Write Store   │──────────►│   Read Store   │
     │   (Source)     │  (Sync)   │  (Optimized)   │
     └────────────────┘           └────────────────┘
```

### Commands and Queries

```python
# Commands (Write)
@dataclass(frozen=True)
class PlaceOrderCommand:
    customer_id: CustomerId
    items: list[OrderItemData]

class PlaceOrderHandler:
    def __init__(self, orders: OrderRepository, events: EventPublisher):
        self._orders = orders
        self._events = events
    
    def handle(self, command: PlaceOrderCommand) -> OrderId:
        order = Order.create(command.customer_id)
        for item in command.items:
            order.add_line(item.product_id, item.quantity)
        order.place()
        self._orders.save(order)
        self._events.publish(order.collect_events())
        return order.id

# Queries (Read)
@dataclass(frozen=True)
class GetOrderDetailsQuery:
    order_id: str

@dataclass
class OrderDetailsView:
    order_id: str
    customer_name: str
    items: list[OrderItemView]
    total: Decimal
    status: str

class GetOrderDetailsHandler:
    def __init__(self, read_db: ReadDatabase):
        self._db = read_db
    
    def handle(self, query: GetOrderDetailsQuery) -> OrderDetailsView | None:
        # Direct optimized query, no domain model
        return self._db.query("""
            SELECT o.id, c.name, o.total, o.status
            FROM order_views o
            JOIN customer_views c ON o.customer_id = c.id
            WHERE o.id = :order_id
        """, order_id=query.order_id)
```

### Read Model Synchronization

```python
# Event handler to update read model
class OrderProjection:
    def __init__(self, read_db: ReadDatabase):
        self._db = read_db
    
    @handles(OrderPlaced)
    def on_order_placed(self, event: OrderPlaced) -> None:
        self._db.execute("""
            INSERT INTO order_views (id, customer_id, total, status, placed_at)
            VALUES (:id, :customer_id, :total, 'placed', :placed_at)
        """, id=event.order_id, customer_id=event.customer_id, 
            total=event.total, placed_at=event.occurred_at)
    
    @handles(OrderShipped)
    def on_order_shipped(self, event: OrderShipped) -> None:
        self._db.execute("""
            UPDATE order_views SET status = 'shipped', tracking = :tracking
            WHERE id = :id
        """, id=event.order_id, tracking=event.tracking_number)
```

---

## Event Sourcing

Store state as sequence of events, not current snapshot.

```
Traditional:                    Event Sourced:
┌─────────────────┐            ┌─────────────────┐
│ Order           │            │ Event Store     │
│ ─────────────── │            │ ─────────────── │
│ id: 123         │            │ OrderCreated    │
│ status: shipped │            │ ItemAdded       │
│ total: $100     │            │ ItemAdded       │
│ items: [...]    │            │ OrderPlaced     │
└─────────────────┘            │ PaymentReceived │
                               │ OrderShipped    │
     Current state             └─────────────────┘
     (lossy)                      Full history
                                  (lossless)
```

### Event-Sourced Aggregate

```python
class Order:
    def __init__(self):
        self._id: OrderId | None = None
        self._status: OrderStatus = OrderStatus.DRAFT
        self._lines: list[OrderLine] = []
        self._uncommitted_events: list[DomainEvent] = []
    
    # Command methods raise events
    def place(self) -> None:
        if self._status != OrderStatus.DRAFT:
            raise InvalidStateError()
        self._apply(OrderPlaced(order_id=self._id, total=self.total))
    
    # Apply methods update state from events
    def _apply(self, event: DomainEvent) -> None:
        self._uncommitted_events.append(event)
        self._mutate(event)
    
    def _mutate(self, event: DomainEvent) -> None:
        match event:
            case OrderCreated(order_id=oid, customer_id=cid):
                self._id = oid
                self._customer_id = cid
            case OrderPlaced():
                self._status = OrderStatus.PLACED
            case ItemAdded(product_id=pid, quantity=qty, price=price):
                self._lines.append(OrderLine(pid, qty, price))
    
    # Reconstitute from event history
    @classmethod
    def from_events(cls, events: list[DomainEvent]) -> "Order":
        order = cls()
        for event in events:
            order._mutate(event)
        return order
    
    def uncommitted_events(self) -> list[DomainEvent]:
        return self._uncommitted_events.copy()
    
    def clear_events(self) -> None:
        self._uncommitted_events.clear()
```

### Event Store Repository

```python
class EventSourcedOrderRepository(OrderRepository):
    def __init__(self, event_store: EventStore):
        self._store = event_store
    
    def find_by_id(self, order_id: OrderId) -> Order | None:
        events = self._store.load_stream(f"order-{order_id}")
        if not events:
            return None
        return Order.from_events(events)
    
    def save(self, order: Order) -> None:
        events = order.uncommitted_events()
        self._store.append_to_stream(f"order-{order.id}", events)
        order.clear_events()
```

---

## Pattern Combinations

### Hexagonal + CQRS

```
                ┌─────────────────────────────────────────┐
                │              Domain Core                │
                │  ┌─────────────────┐ ┌───────────────┐  │
   Commands ───►│  │ Command Ports   │ │  Query Ports  │──┼──► Queries
                │  │ (Write Model)   │ │ (Read Model)  │  │
                │  └────────┬────────┘ └───────┬───────┘  │
                └───────────┼──────────────────┼──────────┘
                            │                  │
                 ┌──────────┴──────┐ ┌─────────┴─────────┐
                 │ Write Adapters  │ │  Read Adapters    │
                 │ (SQL, Events)   │ │  (Cache, Search)  │
                 └─────────────────┘ └───────────────────┘
```

### Hexagonal + CQRS + Event Sourcing

```
Commands ──► Command Handler ──► Aggregate ──► Event Store
                                     │              │
                                     │              ▼
                                     │         Projections
                                     │              │
                                     │              ▼
Queries  ◄── Query Handler  ◄────────┴───────  Read Store
```

---

## Selection Guide

| Requirement | Recommended Pattern |
|-------------|---------------------|
| Simple CRUD, low complexity | Layered |
| Testable domain isolation | Hexagonal |
| Multiple delivery mechanisms | Hexagonal or Clean |
| Complex business rules | Clean Architecture |
| Different read/write needs | CQRS |
| High read scalability | CQRS with read replicas |
| Complete audit trail | Event Sourcing |
| Temporal queries | Event Sourcing |
| Debugging complex state | Event Sourcing |
| Event-driven architecture | Event Sourcing + CQRS |

### Complexity Trade-offs

```
Complexity:  Low ◄─────────────────────────────────────► High

Layered ──► Hexagonal ──► Clean ──► CQRS ──► Event Sourcing

Benefits:   Simplicity   Testability  Flexibility  Scalability  Auditability
```