# Strategic Design

Strategic design focuses on the big picture: how to decompose a large system into bounded contexts and manage their relationships.

## Table of Contents
1. [Subdomains](#subdomains)
2. [Bounded Contexts](#bounded-contexts)
3. [Context Mapping](#context-mapping)
4. [Ubiquitous Language](#ubiquitous-language)
5. [Strategic Design Process](#strategic-design-process)

---

## Subdomains

Subdomains represent areas of the business. Classify by business value:

| Type | Description | Investment | Example |
|------|-------------|------------|---------|
| **Core** | Competitive advantage, unique to business | High - custom build, best talent | Pricing algorithm, recommendation engine |
| **Supporting** | Necessary but not differentiating | Medium - can outsource some | Customer onboarding, reporting |
| **Generic** | Common across industries | Low - buy/use OSS | Authentication, email sending, payments |

**Discovery questions:**
- What makes this business unique?
- What would competitors copy if they could?
- What could we buy off-the-shelf?

---

## Bounded Contexts

A bounded context is a linguistic and model boundary where a domain model applies consistently.

### Key Principles

1. **One model per context**: `Customer` in Sales ≠ `Customer` in Shipping
2. **Explicit boundaries**: Clear interfaces between contexts
3. **Team ownership**: Ideally one team per context
4. **Independent deployment**: Each context can evolve separately

### Identifying Boundaries

Look for:
- **Language shifts**: Same term means different things
- **Model conflicts**: Incompatible representations
- **Team boundaries**: Different groups own different areas
- **Process boundaries**: Distinct business workflows

### Example: E-commerce

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Catalog     │  │     Orders      │  │    Shipping     │
│                 │  │                 │  │                 │
│ Product         │  │ Order           │  │ Shipment        │
│ Category        │  │ OrderLine       │  │ Package         │
│ Price           │  │ Customer (ref)  │  │ Address         │
│ Inventory       │  │ Payment         │  │ Carrier         │
└─────────────────┘  └─────────────────┘  └─────────────────┘
        │                    │                    │
        └────────────────────┴────────────────────┘
                    Context Map
```

---

## Context Mapping

Context maps document relationships between bounded contexts.

### Relationship Patterns

| Pattern | Description | When to Use |
|---------|-------------|-------------|
| **Shared Kernel** | Shared subset of model between contexts | Tight collaboration, same team |
| **Customer-Supplier** | Upstream provides, downstream consumes | Clear dependency direction |
| **Conformist** | Downstream adopts upstream model as-is | No influence over upstream |
| **Anti-Corruption Layer (ACL)** | Translation layer to protect domain | Integrating with legacy/external |
| **Open Host Service (OHS)** | Published API for multiple consumers | Platform/service provider |
| **Published Language** | Shared schema (XML, JSON, Protobuf) | Industry standards, APIs |
| **Separate Ways** | No integration, duplicate if needed | Independence more valuable |
| **Partnership** | Two contexts evolve together | Mutual dependency, same goals |

### Context Map Notation

```
┌───────────────┐         ┌───────────────┐
│   Upstream    │         │  Downstream   │
│   Context     │ ──U/D─► │   Context     │
└───────────────┘         └───────────────┘

U = Upstream (supplier)
D = Downstream (consumer)
ACL = Anti-Corruption Layer
OHS = Open Host Service
PL = Published Language
CF = Conformist
SK = Shared Kernel
```

### Example Context Map

```
                    ┌─────────────────────┐
                    │    Identity &       │
                    │    Access (Generic) │
                    └──────────┬──────────┘
                               │ OHS/PL
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     Catalog     │  │     Orders      │  │    Shipping     │
│     (Core)      │  │     (Core)      │  │  (Supporting)   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         │    U/D             │    U/D             │
         └──────────►─────────┴──────────►─────────┘
                              │
                              │ ACL
                              ▼
                    ┌─────────────────────┐
                    │   Legacy ERP        │
                    │   (External)        │
                    └─────────────────────┘
```

---

## Ubiquitous Language

A shared vocabulary between developers and domain experts within a bounded context.

### Building the Language

1. **Extract from conversations**: Listen to how domain experts speak
2. **Document in glossary**: Maintain living dictionary per context
3. **Use in code**: Class names, methods, variables match domain terms
4. **Refine continuously**: Language evolves with understanding

### Glossary Template

```markdown
## Orders Context - Ubiquitous Language

| Term | Definition | Examples |
|------|------------|----------|
| Order | A customer's request to purchase items | Order #12345 |
| Order Line | Single item entry within an order | "2x Blue Widget" |
| Fulfillment | Process of preparing order for delivery | Picking, packing |
| Backorder | Order for out-of-stock item | Will ship when available |
| Cancel | Customer-initiated order termination | Refund issued |
| Void | System-initiated order termination | Payment failed |
```

### Code Alignment

```python
# BAD: Technical terms
class OrderDTO:
    def process_record(self): ...
    def update_status_flag(self): ...

# GOOD: Ubiquitous language
class Order:
    def place(self): ...
    def cancel(self, reason: CancellationReason): ...
    def fulfill(self): ...
```

---

## Strategic Design Process

### Step 1: Domain Discovery

1. Interview domain experts
2. Map business capabilities
3. Identify core vs supporting vs generic

### Step 2: Context Identification

1. Look for linguistic boundaries
2. Identify model conflicts
3. Map to team structure
4. Define explicit boundaries

### Step 3: Context Mapping

1. Document all contexts
2. Identify relationships
3. Choose integration patterns
4. Define interfaces

### Step 4: Language Development

1. Create glossary per context
2. Validate with domain experts
3. Embed in code
4. Review and refine

### Deliverables Checklist

- [ ] Subdomain classification (Core/Supporting/Generic)
- [ ] Bounded context diagram
- [ ] Context map with relationship patterns
- [ ] Ubiquitous language glossary per context
- [ ] Integration contracts between contexts