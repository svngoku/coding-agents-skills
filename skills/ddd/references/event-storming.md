# Event Storming

Collaborative workshop technique for exploring complex business domains.

## Table of Contents
1. [Overview](#overview)
2. [Sticky Note Legend](#sticky-note-legend)
3. [Workshop Phases](#workshop-phases)
4. [Big Picture Event Storming](#big-picture-event-storming)
5. [Process Modeling](#process-modeling)
6. [Software Design](#software-design)
7. [Facilitation Tips](#facilitation-tips)
8. [Documentation Templates](#documentation-templates)

---

## Overview

Event Storming explores business domains by focusing on **what happens** (events) rather than data structures.

**Participants**: Developers, domain experts, stakeholders, product owners
**Duration**: 2-4 hours for Big Picture, 1-2 days for full Design Level
**Materials**: Large wall space, sticky notes (multiple colors), markers

---

## Sticky Note Legend

| Color | Element | Description | Example |
|-------|---------|-------------|---------|
| 🟧 Orange | **Domain Event** | Something that happened (past tense) | "Order Placed" |
| 🟦 Blue | **Command** | Action that triggers event | "Place Order" |
| 🟨 Yellow | **Actor/User** | Person or system that issues command | "Customer" |
| 🟪 Purple | **Policy** | Reactive logic ("When X, then Y") | "When Order Placed, Reserve Inventory" |
| 🟩 Green | **Read Model** | Information needed to make decision | "Available Inventory" |
| 🟥 Red/Pink | **Hotspot** | Problem, question, or conflict | "What if payment fails?" |
| 🟫 Tan/Beige | **Aggregate** | Cluster of related events/commands | "Order" |
| ⬜ White | **External System** | System outside bounded context | "Payment Gateway" |

---

## Workshop Phases

### Phase 1: Chaotic Exploration (30-60 min)
1. Everyone writes domain events on orange stickies
2. Place on wall in rough time order (left to right)
3. No discussion yet - just capture everything
4. Duplicates are fine

### Phase 2: Enforce Timeline (30 min)
1. Arrange events in chronological order
2. Identify parallel flows (stack vertically)
3. Mark pivotal events (key moments)
4. Add hotspots for questions/conflicts

### Phase 3: Add Commands & Actors (30-45 min)
1. For each event, ask "What caused this?"
2. Add blue command sticky before event
3. Add yellow actor sticky to command
4. Connect with arrows

### Phase 4: Identify Aggregates (30 min)
1. Group related events/commands
2. Name the aggregate (tan sticky)
3. Draw boundaries
4. Note aggregate relationships

### Phase 5: Identify Bounded Contexts (30 min)
1. Look for language boundaries
2. Group aggregates into contexts
3. Name contexts
4. Identify context relationships

---

## Big Picture Event Storming

Fast exploration of entire business domain. Output: shared understanding, bounded contexts, hot spots.

### Process

```
Step 1: Capture Events
┌─────────────────────────────────────────────────────────────────┐
│ 🟧 User      🟧 Order    🟧 Payment   🟧 Order    🟧 Order     │
│    Registered   Created    Received     Shipped     Delivered   │
└─────────────────────────────────────────────────────────────────┘
                         Time →

Step 2: Add Pivotal Events (mark with vertical line)
┌─────────────────────────────────────────────────────────────────┐
│ 🟧 User      │🟧 Order   │🟧 Payment  │🟧 Order   🟧 Order     │
│    Registered│   Placed  │   Received │   Shipped    Delivered  │
└─────────────────────────────────────────────────────────────────┘
              Pivotal     Pivotal      Pivotal

Step 3: Identify Swimlanes (parallel processes)
┌─────────────────────────────────────────────────────────────────┐
│ Sales:    🟧 Order Placed → 🟧 Order Confirmed                  │
│ Payment:  🟧 Payment Initiated → 🟧 Payment Received            │
│ Warehouse:🟧 Picking Started → 🟧 Order Packed → 🟧 Shipped     │
└─────────────────────────────────────────────────────────────────┘

Step 4: Mark Hotspots
┌─────────────────────────────────────────────────────────────────┐
│ 🟧 Order Placed → 🟥 What if inventory unavailable? → 🟧 ...    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Process Modeling

Add commands, actors, policies, and read models.

### Complete Notation

```
   🟨 Actor                   🟪 Policy
      │                          │
      ▼                          ▼
   🟦 Command ──► 🟫 Aggregate ──► 🟧 Event ──► 🟪 Policy
      │                                            │
      │                                            ▼
   🟩 Read Model                              🟦 Command
   (Information needed                        (Triggered by policy)
    to make decision)
```

### Example: Order Placement

```
🟨 Customer
     │
     │ uses
     ▼
🟩 Shopping Cart                    🟩 Available Credit
     │                                     │
     │ views                               │ checks
     ▼                                     ▼
🟦 Place Order ────────────────────► 🟫 Order ────► 🟧 Order Placed
                                                          │
                                                          │ triggers
                                                          ▼
                                                    🟪 When Order Placed,
                                                       Reserve Inventory
                                                          │
                                                          ▼
                                                    🟦 Reserve Stock
                                                          │
                                                          ▼
                                    ⬜ Inventory ──► 🟧 Stock Reserved
                                       System
```

---

## Software Design

Translate Event Storming to DDD building blocks.

### Mapping to Code

| Event Storming | DDD Building Block |
|----------------|-------------------|
| 🟫 Aggregate | Aggregate Root |
| 🟧 Domain Event | Domain Event class |
| 🟦 Command | Command object |
| 🟪 Policy | Domain Service or Event Handler |
| 🟩 Read Model | Query/View Model |
| ⬜ External System | Anti-Corruption Layer |
| Swimlane boundary | Bounded Context |

### From Stickies to Code

```
Event Storming:
🟦 Place Order → 🟫 Order → 🟧 Order Placed → 🟪 Reserve Inventory

Code Structure:
commands/
  place_order.py          # 🟦 PlaceOrderCommand
domain/
  order.py                # 🟫 Order (Aggregate)
events/
  order_placed.py         # 🟧 OrderPlaced
policies/
  inventory_policy.py     # 🟪 ReserveInventoryPolicy
```

```python
# Direct translation
@dataclass
class PlaceOrderCommand:  # 🟦
    customer_id: str
    items: list[OrderItem]

class Order:  # 🟫
    def place(self) -> None:
        self._events.append(OrderPlaced(...))  # 🟧

class OrderPlaced(DomainEvent):  # 🟧
    order_id: OrderId
    customer_id: CustomerId

class InventoryPolicy:  # 🟪
    @handles(OrderPlaced)
    def on_order_placed(self, event: OrderPlaced) -> None:
        self._inventory.reserve(event.order_id, event.items)
```

---

## Facilitation Tips

### Before Workshop
- [ ] Book large room with empty wall space
- [ ] Prepare sticky notes (all colors)
- [ ] Invite domain experts AND developers
- [ ] Send brief overview to participants
- [ ] Prepare domain question list

### During Workshop
- Keep energy high, discourage sitting
- Encourage everyone to write stickies
- Domain experts write in business language
- Developers ask clarifying questions
- Capture ALL hotspots, don't solve immediately
- Take photos frequently

### Common Problems

| Problem | Solution |
|---------|----------|
| One person dominates | Hand them the marker, make them facilitate |
| Too much detail early | "We'll come back to that" - add hotspot |
| Analysis paralysis | Time-box each phase |
| Arguments over terminology | Perfect! You found a context boundary |
| "It depends" answers | Great - capture both paths |

---

## Documentation Templates

### Event Catalog

```markdown
## Events in [Context Name]

### OrderPlaced
- **Trigger**: Customer completes checkout
- **Data**: order_id, customer_id, items[], total, placed_at
- **Policies triggered**: 
  - Reserve Inventory
  - Send Confirmation Email
  - Start Payment Processing
- **Aggregate**: Order

### OrderShipped
- **Trigger**: Warehouse marks order as shipped
- **Data**: order_id, tracking_number, carrier, shipped_at
- **Policies triggered**:
  - Send Shipping Notification
  - Update Delivery Estimate
- **Aggregate**: Order
```

### Aggregate Documentation

```markdown
## Order Aggregate

### Commands
| Command | Actor | Preconditions | Events |
|---------|-------|---------------|--------|
| Place Order | Customer | Cart not empty, valid payment | Order Placed |
| Cancel Order | Customer | Status = Pending | Order Cancelled |
| Ship Order | Warehouse | Status = Paid | Order Shipped |

### Invariants
- Order must have at least one line item
- Total must match sum of line items
- Cannot modify after shipping

### Events Emitted
- OrderPlaced
- OrderCancelled
- OrderShipped
- OrderDelivered
```

### Context Map Documentation

```markdown
## Context: Order Management

### Upstream Dependencies
- **Identity Context** (OHS/PL): User authentication
- **Catalog Context** (Customer/Supplier): Product information

### Downstream Consumers  
- **Shipping Context** (Customer/Supplier): Fulfillment
- **Analytics Context** (Conformist): Reporting

### External Systems
- **Payment Gateway** (ACL): Stripe integration
- **Tax Service** (ACL): Avalara integration

### Ubiquitous Language
| Term | Definition |
|------|------------|
| Order | Customer's intent to purchase items |
| Line Item | Single product entry in order |
| Fulfillment | Process of shipping order to customer |
```

### Bounded Context Canvas

```
┌─────────────────────────────────────────────────────────────────┐
│ BOUNDED CONTEXT: Order Management                               │
├─────────────────────────────────────────────────────────────────┤
│ PURPOSE: Handle customer orders from placement to delivery      │
├──────────────────────────────┬──────────────────────────────────┤
│ INBOUND COMMUNICATION        │ OUTBOUND COMMUNICATION           │
│ • PlaceOrder (sync)          │ • OrderPlaced → Inventory        │
│ • CancelOrder (sync)         │ • OrderShipped → Notifications   │
│ • GetOrderStatus (sync)      │ • OrderCompleted → Analytics     │
├──────────────────────────────┼──────────────────────────────────┤
│ UBIQUITOUS LANGUAGE          │ BUSINESS RULES                   │
│ • Order, LineItem            │ • Min order value: $10           │
│ • Fulfillment, Shipment      │ • Max items per order: 100       │
│ • Backorder, Cancellation    │ • Cancel window: 24 hours        │
├──────────────────────────────┴──────────────────────────────────┤
│ KEY AGGREGATES: Order, OrderLine                                │
├─────────────────────────────────────────────────────────────────┤
│ TEAM: Order Squad (4 devs, 1 PM)                                │
└─────────────────────────────────────────────────────────────────┘
```