# TypeScript DDD Patterns

Implementations using popular TypeScript frameworks: NestJS, TypeORM, Prisma.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Value Objects](#value-objects)
3. [Entities and Aggregates](#entities-and-aggregates)
4. [Domain Events](#domain-events)
5. [Repositories with TypeORM](#repositories-with-typeorm)
6. [Repositories with Prisma](#repositories-with-prisma)
7. [Application Services](#application-services)
8. [NestJS Integration](#nestjs-integration)
9. [Testing Patterns](#testing-patterns)

---

## Project Setup

### Recommended Structure

```
src/
├── domain/
│   ├── model/
│   │   ├── order.ts           # Aggregate
│   │   ├── order-line.ts      # Entity
│   │   └── value-objects/
│   │       ├── money.ts
│   │       ├── order-id.ts
│   │       └── index.ts
│   ├── event/
│   │   ├── domain-event.ts
│   │   └── order-events.ts
│   ├── repository/
│   │   └── order.repository.ts  # Interface
│   └── service/
│       └── pricing.service.ts
├── application/
│   ├── command/
│   │   └── place-order.handler.ts
│   ├── query/
│   │   └── get-order.handler.ts
│   └── dto/
│       └── order.dto.ts
├── infrastructure/
│   ├── persistence/
│   │   ├── typeorm/
│   │   │   ├── entities/
│   │   │   └── order.repository.ts
│   │   └── prisma/
│   │       ├── schema.prisma
│   │       └── order.repository.ts
│   └── messaging/
│       └── event-bus.ts
└── interface/
    └── http/
        ├── controllers/
        │   └── order.controller.ts
        └── dto/
            └── create-order.dto.ts
```

### Dependencies (package.json)

```json
{
  "dependencies": {
    "@nestjs/common": "^10.0.0",
    "@nestjs/core": "^10.0.0",
    "@nestjs/cqrs": "^10.0.0",
    "@prisma/client": "^5.0.0",
    "typeorm": "^0.3.0",
    "uuid": "^9.0.0"
  },
  "devDependencies": {
    "@types/uuid": "^9.0.0",
    "typescript": "^5.0.0",
    "jest": "^29.0.0"
  }
}
```

---

## Value Objects

```typescript
// domain/model/value-objects/money.ts
export class Money {
  private constructor(
    public readonly amount: number,
    public readonly currency: string
  ) {
    if (amount < 0) {
      throw new Error('Amount cannot be negative');
    }
    if (currency.length !== 3) {
      throw new Error('Currency must be 3-letter ISO code');
    }
  }

  static create(amount: number, currency: string = 'USD'): Money {
    return new Money(
      Math.round(amount * 100) / 100,
      currency.toUpperCase()
    );
  }

  static zero(currency: string = 'USD'): Money {
    return new Money(0, currency);
  }

  add(other: Money): Money {
    this.assertSameCurrency(other);
    return new Money(this.amount + other.amount, this.currency);
  }

  subtract(other: Money): Money {
    this.assertSameCurrency(other);
    return new Money(this.amount - other.amount, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(this.amount * factor, this.currency);
  }

  equals(other: Money): boolean {
    return this.amount === other.amount && this.currency === other.currency;
  }

  private assertSameCurrency(other: Money): void {
    if (this.currency !== other.currency) {
      throw new Error(`Currency mismatch: ${this.currency} vs ${other.currency}`);
    }
  }
}

// domain/model/value-objects/order-id.ts
import { v4 as uuidv4 } from 'uuid';

export class OrderId {
  private constructor(public readonly value: string) {
    if (!value || value.trim() === '') {
      throw new Error('OrderId cannot be empty');
    }
  }

  static create(value: string): OrderId {
    return new OrderId(value);
  }

  static generate(): OrderId {
    return new OrderId(uuidv4());
  }

  equals(other: OrderId): boolean {
    return this.value === other.value;
  }

  toString(): string {
    return this.value;
  }
}

// domain/model/value-objects/customer-id.ts
export class CustomerId {
  private constructor(public readonly value: string) {}

  static create(value: string): CustomerId {
    return new CustomerId(value);
  }

  equals(other: CustomerId): boolean {
    return this.value === other.value;
  }

  toString(): string {
    return this.value;
  }
}

// domain/model/value-objects/address.ts
export class Address {
  private constructor(
    public readonly street: string,
    public readonly city: string,
    public readonly state: string,
    public readonly postalCode: string,
    public readonly country: string
  ) {}

  static create(props: {
    street: string;
    city: string;
    state: string;
    postalCode: string;
    country?: string;
  }): Address {
    return new Address(
      props.street,
      props.city,
      props.state,
      props.postalCode,
      props.country ?? 'US'
    );
  }

  equals(other: Address): boolean {
    return (
      this.street === other.street &&
      this.city === other.city &&
      this.state === other.state &&
      this.postalCode === other.postalCode &&
      this.country === other.country
    );
  }

  formatSingleLine(): string {
    return `${this.street}, ${this.city}, ${this.state} ${this.postalCode}`;
  }
}
```

---

## Entities and Aggregates

```typescript
// domain/model/order-line.ts
import { Money } from './value-objects/money';

export class OrderLine {
  constructor(
    public readonly productId: string,
    public readonly quantity: number,
    public readonly unitPrice: Money
  ) {
    if (quantity <= 0) {
      throw new Error('Quantity must be positive');
    }
  }

  get subtotal(): Money {
    return this.unitPrice.multiply(this.quantity);
  }
}

// domain/model/order.ts
import { OrderId, CustomerId, Money } from './value-objects';
import { OrderLine } from './order-line';
import { DomainEvent } from '../event/domain-event';
import { OrderPlaced, OrderCancelled } from '../event/order-events';

export enum OrderStatus {
  Draft = 'draft',
  Placed = 'placed',
  Paid = 'paid',
  Shipped = 'shipped',
  Delivered = 'delivered',
  Cancelled = 'cancelled',
}

export class Order {
  private _events: DomainEvent[] = [];

  private constructor(
    public readonly id: OrderId,
    public readonly customerId: CustomerId,
    private _status: OrderStatus,
    private _lines: OrderLine[],
    private _placedAt: Date | null
  ) {}

  // Factory
  static create(customerId: CustomerId): Order {
    return new Order(
      OrderId.generate(),
      customerId,
      OrderStatus.Draft,
      [],
      null
    );
  }

  // Reconstitution (from persistence)
  static reconstitute(props: {
    id: OrderId;
    customerId: CustomerId;
    status: OrderStatus;
    lines: OrderLine[];
    placedAt: Date | null;
  }): Order {
    return new Order(
      props.id,
      props.customerId,
      props.status,
      props.lines,
      props.placedAt
    );
  }

  // Getters
  get status(): OrderStatus {
    return this._status;
  }

  get lines(): ReadonlyArray<OrderLine> {
    return [...this._lines];
  }

  get placedAt(): Date | null {
    return this._placedAt;
  }

  get total(): Money {
    if (this._lines.length === 0) {
      return Money.zero();
    }
    return this._lines.reduce(
      (sum, line) => sum.add(line.subtotal),
      Money.zero()
    );
  }

  get itemCount(): number {
    return this._lines.reduce((sum, line) => sum + line.quantity, 0);
  }

  // Commands
  addLine(productId: string, quantity: number, unitPrice: Money): void {
    this.assertDraft();
    this.assertProductNotExists(productId);

    this._lines.push(new OrderLine(productId, quantity, unitPrice));
  }

  removeLine(productId: string): void {
    this.assertDraft();
    this._lines = this._lines.filter((l) => l.productId !== productId);
  }

  place(): void {
    this.assertDraft();
    if (this._lines.length === 0) {
      throw new EmptyOrderError('Cannot place order without items');
    }

    this._status = OrderStatus.Placed;
    this._placedAt = new Date();

    this._events.push(
      new OrderPlaced(this.id, this.customerId, this.total, this._placedAt)
    );
  }

  cancel(reason: string): void {
    if (
      this._status === OrderStatus.Shipped ||
      this._status === OrderStatus.Delivered
    ) {
      throw new InvalidOrderStateError(
        'Cannot cancel shipped/delivered order'
      );
    }

    this._status = OrderStatus.Cancelled;
    this._events.push(new OrderCancelled(this.id, reason));
  }

  // Events
  collectEvents(): DomainEvent[] {
    const events = [...this._events];
    this._events = [];
    return events;
  }

  // Invariants
  private assertDraft(): void {
    if (this._status !== OrderStatus.Draft) {
      throw new InvalidOrderStateError('Can only modify draft orders');
    }
  }

  private assertProductNotExists(productId: string): void {
    if (this._lines.some((l) => l.productId === productId)) {
      throw new DuplicateProductError(`Product ${productId} already in order`);
    }
  }
}

// Exceptions
export class OrderError extends Error {}
export class InvalidOrderStateError extends OrderError {}
export class EmptyOrderError extends OrderError {}
export class DuplicateProductError extends OrderError {}
```

---

## Domain Events

```typescript
// domain/event/domain-event.ts
import { v4 as uuidv4 } from 'uuid';

export abstract class DomainEvent {
  public readonly eventId: string;
  public readonly occurredAt: Date;

  constructor() {
    this.eventId = uuidv4();
    this.occurredAt = new Date();
  }
}

// domain/event/order-events.ts
import { DomainEvent } from './domain-event';
import { OrderId, CustomerId, Money } from '../model/value-objects';

export class OrderPlaced extends DomainEvent {
  constructor(
    public readonly orderId: OrderId,
    public readonly customerId: CustomerId,
    public readonly total: Money,
    public readonly placedAt: Date
  ) {
    super();
  }
}

export class OrderCancelled extends DomainEvent {
  constructor(
    public readonly orderId: OrderId,
    public readonly reason: string
  ) {
    super();
  }
}

export class OrderShipped extends DomainEvent {
  constructor(
    public readonly orderId: OrderId,
    public readonly trackingNumber: string,
    public readonly carrier: string
  ) {
    super();
  }
}
```

---

## Repositories with TypeORM

```typescript
// domain/repository/order.repository.ts
import { Order } from '../model/order';
import { OrderId, CustomerId } from '../model/value-objects';

export interface OrderRepository {
  findById(orderId: OrderId): Promise<Order | null>;
  save(order: Order): Promise<void>;
  findByCustomer(customerId: CustomerId): Promise<Order[]>;
}

export const ORDER_REPOSITORY = Symbol('OrderRepository');

// infrastructure/persistence/typeorm/entities/order.entity.ts
import {
  Entity,
  PrimaryColumn,
  Column,
  OneToMany,
  CreateDateColumn,
} from 'typeorm';
import { OrderLineEntity } from './order-line.entity';

@Entity('orders')
export class OrderEntity {
  @PrimaryColumn('uuid')
  id: string;

  @Column('uuid')
  customerId: string;

  @Column({ length: 20 })
  status: string;

  @Column({ type: 'timestamp', nullable: true })
  placedAt: Date | null;

  @OneToMany(() => OrderLineEntity, (line) => line.order, {
    cascade: true,
    eager: true,
  })
  lines: OrderLineEntity[];
}

// infrastructure/persistence/typeorm/entities/order-line.entity.ts
import { Entity, PrimaryGeneratedColumn, Column, ManyToOne } from 'typeorm';
import { OrderEntity } from './order.entity';

@Entity('order_lines')
export class OrderLineEntity {
  @PrimaryGeneratedColumn()
  id: number;

  @Column('uuid')
  productId: string;

  @Column('int')
  quantity: number;

  @Column('decimal', { precision: 10, scale: 2 })
  unitPriceAmount: number;

  @Column({ length: 3 })
  unitPriceCurrency: string;

  @ManyToOne(() => OrderEntity, (order) => order.lines)
  order: OrderEntity;
}

// infrastructure/persistence/typeorm/order.repository.ts
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';

import { OrderRepository } from '../../../domain/repository/order.repository';
import { Order, OrderStatus } from '../../../domain/model/order';
import { OrderLine } from '../../../domain/model/order-line';
import { OrderId, CustomerId, Money } from '../../../domain/model/value-objects';
import { OrderEntity } from './entities/order.entity';
import { OrderLineEntity } from './entities/order-line.entity';

@Injectable()
export class TypeOrmOrderRepository implements OrderRepository {
  constructor(
    @InjectRepository(OrderEntity)
    private readonly repo: Repository<OrderEntity>
  ) {}

  async findById(orderId: OrderId): Promise<Order | null> {
    const entity = await this.repo.findOne({
      where: { id: orderId.value },
      relations: ['lines'],
    });
    return entity ? this.toDomain(entity) : null;
  }

  async save(order: Order): Promise<void> {
    const entity = this.toEntity(order);
    await this.repo.save(entity);
  }

  async findByCustomer(customerId: CustomerId): Promise<Order[]> {
    const entities = await this.repo.find({
      where: { customerId: customerId.value },
      relations: ['lines'],
    });
    return entities.map((e) => this.toDomain(e));
  }

  private toDomain(entity: OrderEntity): Order {
    return Order.reconstitute({
      id: OrderId.create(entity.id),
      customerId: CustomerId.create(entity.customerId),
      status: entity.status as OrderStatus,
      placedAt: entity.placedAt,
      lines: entity.lines.map(
        (l) =>
          new OrderLine(
            l.productId,
            l.quantity,
            Money.create(l.unitPriceAmount, l.unitPriceCurrency)
          )
      ),
    });
  }

  private toEntity(order: Order): OrderEntity {
    const entity = new OrderEntity();
    entity.id = order.id.value;
    entity.customerId = order.customerId.value;
    entity.status = order.status;
    entity.placedAt = order.placedAt;
    entity.lines = order.lines.map((l) => {
      const line = new OrderLineEntity();
      line.productId = l.productId;
      line.quantity = l.quantity;
      line.unitPriceAmount = l.unitPrice.amount;
      line.unitPriceCurrency = l.unitPrice.currency;
      return line;
    });
    return entity;
  }
}
```

---

## Repositories with Prisma

```prisma
// infrastructure/persistence/prisma/schema.prisma
model Order {
  id         String      @id @default(uuid())
  customerId String
  status     String
  placedAt   DateTime?
  lines      OrderLine[]
}

model OrderLine {
  id                 Int     @id @default(autoincrement())
  orderId            String
  productId          String
  quantity           Int
  unitPriceAmount    Decimal @db.Decimal(10, 2)
  unitPriceCurrency  String  @db.VarChar(3)
  order              Order   @relation(fields: [orderId], references: [id])
}
```

```typescript
// infrastructure/persistence/prisma/order.repository.ts
import { Injectable } from '@nestjs/common';
import { PrismaService } from './prisma.service';

import { OrderRepository } from '../../../domain/repository/order.repository';
import { Order, OrderStatus } from '../../../domain/model/order';
import { OrderLine } from '../../../domain/model/order-line';
import { OrderId, CustomerId, Money } from '../../../domain/model/value-objects';

@Injectable()
export class PrismaOrderRepository implements OrderRepository {
  constructor(private readonly prisma: PrismaService) {}

  async findById(orderId: OrderId): Promise<Order | null> {
    const data = await this.prisma.order.findUnique({
      where: { id: orderId.value },
      include: { lines: true },
    });
    return data ? this.toDomain(data) : null;
  }

  async save(order: Order): Promise<void> {
    await this.prisma.order.upsert({
      where: { id: order.id.value },
      create: {
        id: order.id.value,
        customerId: order.customerId.value,
        status: order.status,
        placedAt: order.placedAt,
        lines: {
          create: order.lines.map((l) => ({
            productId: l.productId,
            quantity: l.quantity,
            unitPriceAmount: l.unitPrice.amount,
            unitPriceCurrency: l.unitPrice.currency,
          })),
        },
      },
      update: {
        customerId: order.customerId.value,
        status: order.status,
        placedAt: order.placedAt,
        lines: {
          deleteMany: {},
          create: order.lines.map((l) => ({
            productId: l.productId,
            quantity: l.quantity,
            unitPriceAmount: l.unitPrice.amount,
            unitPriceCurrency: l.unitPrice.currency,
          })),
        },
      },
    });
  }

  async findByCustomer(customerId: CustomerId): Promise<Order[]> {
    const data = await this.prisma.order.findMany({
      where: { customerId: customerId.value },
      include: { lines: true },
    });
    return data.map((d) => this.toDomain(d));
  }

  private toDomain(data: any): Order {
    return Order.reconstitute({
      id: OrderId.create(data.id),
      customerId: CustomerId.create(data.customerId),
      status: data.status as OrderStatus,
      placedAt: data.placedAt,
      lines: data.lines.map(
        (l: any) =>
          new OrderLine(
            l.productId,
            l.quantity,
            Money.create(Number(l.unitPriceAmount), l.unitPriceCurrency)
          )
      ),
    });
  }
}
```

---

## Application Services

```typescript
// application/command/place-order.handler.ts
import { CommandHandler, ICommandHandler, EventBus } from '@nestjs/cqrs';
import { Inject } from '@nestjs/common';

import { Order } from '../../domain/model/order';
import { CustomerId, Money } from '../../domain/model/value-objects';
import {
  OrderRepository,
  ORDER_REPOSITORY,
} from '../../domain/repository/order.repository';

export class PlaceOrderCommand {
  constructor(
    public readonly customerId: string,
    public readonly items: Array<{
      productId: string;
      quantity: number;
      unitPrice: number;
      currency?: string;
    }>
  ) {}
}

@CommandHandler(PlaceOrderCommand)
export class PlaceOrderHandler implements ICommandHandler<PlaceOrderCommand> {
  constructor(
    @Inject(ORDER_REPOSITORY)
    private readonly orderRepository: OrderRepository,
    private readonly eventBus: EventBus
  ) {}

  async execute(command: PlaceOrderCommand): Promise<string> {
    const order = Order.create(CustomerId.create(command.customerId));

    for (const item of command.items) {
      order.addLine(
        item.productId,
        item.quantity,
        Money.create(item.unitPrice, item.currency ?? 'USD')
      );
    }

    order.place();
    await this.orderRepository.save(order);

    // Publish events
    for (const event of order.collectEvents()) {
      this.eventBus.publish(event);
    }

    return order.id.value;
  }
}
```

---

## NestJS Integration

```typescript
// interface/http/controllers/order.controller.ts
import { Controller, Post, Body, HttpException, HttpStatus } from '@nestjs/common';
import { CommandBus } from '@nestjs/cqrs';

import { PlaceOrderCommand } from '../../../application/command/place-order.handler';
import { OrderError } from '../../../domain/model/order';

class OrderItemDto {
  productId: string;
  quantity: number;
  unitPrice: number;
  currency?: string;
}

class CreateOrderDto {
  customerId: string;
  items: OrderItemDto[];
}

class CreateOrderResponseDto {
  orderId: string;
}

@Controller('orders')
export class OrderController {
  constructor(private readonly commandBus: CommandBus) {}

  @Post()
  async createOrder(@Body() dto: CreateOrderDto): Promise<CreateOrderResponseDto> {
    try {
      const orderId = await this.commandBus.execute(
        new PlaceOrderCommand(dto.customerId, dto.items)
      );
      return { orderId };
    } catch (error) {
      if (error instanceof OrderError) {
        throw new HttpException(error.message, HttpStatus.BAD_REQUEST);
      }
      throw error;
    }
  }
}

// Module setup
import { Module } from '@nestjs/common';
import { CqrsModule } from '@nestjs/cqrs';
import { TypeOrmModule } from '@nestjs/typeorm';

import { OrderController } from './interface/http/controllers/order.controller';
import { PlaceOrderHandler } from './application/command/place-order.handler';
import { TypeOrmOrderRepository } from './infrastructure/persistence/typeorm/order.repository';
import { OrderEntity } from './infrastructure/persistence/typeorm/entities/order.entity';
import { ORDER_REPOSITORY } from './domain/repository/order.repository';

@Module({
  imports: [CqrsModule, TypeOrmModule.forFeature([OrderEntity])],
  controllers: [OrderController],
  providers: [
    PlaceOrderHandler,
    {
      provide: ORDER_REPOSITORY,
      useClass: TypeOrmOrderRepository,
    },
  ],
})
export class OrderModule {}
```

---

## Testing Patterns

```typescript
// tests/domain/order.spec.ts
import { Order, OrderStatus, EmptyOrderError, InvalidOrderStateError } from '../../domain/model/order';
import { CustomerId, Money } from '../../domain/model/value-objects';
import { OrderPlaced } from '../../domain/event/order-events';

describe('Order', () => {
  describe('create', () => {
    it('creates draft order', () => {
      const order = Order.create(CustomerId.create('cust-123'));

      expect(order.status).toBe(OrderStatus.Draft);
      expect(order.lines).toHaveLength(0);
    });
  });

  describe('addLine', () => {
    it('adds line to order', () => {
      const order = Order.create(CustomerId.create('cust-123'));

      order.addLine('prod-1', 2, Money.create(10));

      expect(order.lines).toHaveLength(1);
      expect(order.total.amount).toBe(20);
    });
  });

  describe('place', () => {
    it('places order and emits event', () => {
      const order = Order.create(CustomerId.create('cust-123'));
      order.addLine('prod-1', 1, Money.create(10));

      order.place();

      expect(order.status).toBe(OrderStatus.Placed);
      const events = order.collectEvents();
      expect(events).toHaveLength(1);
      expect(events[0]).toBeInstanceOf(OrderPlaced);
    });

    it('throws on empty order', () => {
      const order = Order.create(CustomerId.create('cust-123'));

      expect(() => order.place()).toThrow(EmptyOrderError);
    });
  });

  describe('modify placed order', () => {
    it('throws when adding line to placed order', () => {
      const order = Order.create(CustomerId.create('cust-123'));
      order.addLine('prod-1', 1, Money.create(10));
      order.place();

      expect(() => order.addLine('prod-2', 1, Money.create(5))).toThrow(
        InvalidOrderStateError
      );
    });
  });
});

// tests/infrastructure/order.repository.spec.ts
describe('TypeOrmOrderRepository', () => {
  let repository: TypeOrmOrderRepository;
  let dataSource: DataSource;

  beforeAll(async () => {
    dataSource = await createTestDataSource();
    repository = new TypeOrmOrderRepository(
      dataSource.getRepository(OrderEntity)
    );
  });

  afterAll(async () => {
    await dataSource.destroy();
  });

  it('saves and retrieves order', async () => {
    const order = Order.create(CustomerId.create('cust-123'));
    order.addLine('prod-1', 2, Money.create(10));

    await repository.save(order);
    const retrieved = await repository.findById(order.id);

    expect(retrieved).not.toBeNull();
    expect(retrieved!.id.equals(order.id)).toBe(true);
    expect(retrieved!.lines).toHaveLength(1);
  });
});
```