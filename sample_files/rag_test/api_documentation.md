# ACME API Documentation v2.5

## Overview

The ACME API provides programmatic access to ACME Corporation's product catalog, order management,
and inventory systems. This RESTful API uses JSON for request and response bodies.

**Base URL:** `https://api.acmecorp.com/v2`

**Authentication:** All API requests require a valid API key passed in the `X-API-Key` header.

## Rate Limits

| Plan | Requests/Minute | Requests/Day |
|------|-----------------|--------------|
| Free | 60 | 1,000 |
| Basic | 300 | 10,000 |
| Pro | 1,000 | 100,000 |
| Enterprise | Unlimited | Unlimited |

Rate limit headers are included in every response:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when the limit resets

## Authentication

### Getting an API Key

1. Log in to your ACME Developer Portal account at https://developers.acmecorp.com
2. Navigate to "API Keys" in the dashboard
3. Click "Generate New Key"
4. Store your key securely - it will only be shown once

### Using Your API Key

Include the API key in the header of every request:

```
X-API-Key: your_api_key_here
```

## Endpoints

### Products

#### List All Products

```
GET /products
```

**Query Parameters:**
- `page` (integer): Page number for pagination (default: 1)
- `limit` (integer): Items per page, max 100 (default: 20)
- `category` (string): Filter by category
- `min_price` (decimal): Minimum price filter
- `max_price` (decimal): Maximum price filter
- `in_stock` (boolean): Filter by availability

**Response:**
```json
{
  "data": [
    {
      "id": "prod_12345",
      "name": "Super Widget",
      "description": "Our best-selling widget",
      "price": 29.99,
      "currency": "USD",
      "category": "widgets",
      "in_stock": true,
      "quantity_available": 150,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total_items": 150,
    "total_pages": 8
  }
}
```

#### Get Product by ID

```
GET /products/{product_id}
```

**Response:** Returns a single product object.

#### Create Product (Admin only)

```
POST /products
```

**Request Body:**
```json
{
  "name": "New Product",
  "description": "Product description",
  "price": 49.99,
  "category": "gadgets",
  "quantity": 100
}
```

### Orders

#### Create Order

```
POST /orders
```

**Request Body:**
```json
{
  "customer_email": "customer@example.com",
  "shipping_address": {
    "street": "123 Main St",
    "city": "San Jose",
    "state": "CA",
    "zip": "95134",
    "country": "US"
  },
  "items": [
    {
      "product_id": "prod_12345",
      "quantity": 2
    }
  ],
  "payment_method_id": "pm_abc123"
}
```

**Response:**
```json
{
  "id": "ord_67890",
  "status": "pending",
  "total": 59.98,
  "currency": "USD",
  "created_at": "2024-01-20T14:22:00Z",
  "estimated_delivery": "2024-01-25"
}
```

#### Get Order Status

```
GET /orders/{order_id}
```

**Order Statuses:**
- `pending`: Order received, awaiting payment confirmation
- `confirmed`: Payment confirmed, preparing for shipment
- `shipped`: Order has been shipped
- `delivered`: Order delivered to customer
- `cancelled`: Order was cancelled
- `refunded`: Order was refunded

#### List Customer Orders

```
GET /orders?customer_email={email}
```

### Inventory

#### Check Stock

```
GET /inventory/{product_id}
```

**Response:**
```json
{
  "product_id": "prod_12345",
  "quantity_available": 150,
  "quantity_reserved": 12,
  "reorder_point": 50,
  "last_restocked": "2024-01-10T08:00:00Z"
}
```

#### Update Stock (Admin only)

```
PATCH /inventory/{product_id}
```

**Request Body:**
```json
{
  "quantity_adjustment": 50,
  "reason": "Restocked from supplier"
}
```

## Webhooks

ACME API supports webhooks for real-time notifications about order events.

### Available Events

- `order.created`: New order placed
- `order.confirmed`: Payment confirmed
- `order.shipped`: Order shipped
- `order.delivered`: Order delivered
- `order.cancelled`: Order cancelled
- `inventory.low`: Stock below reorder point

### Webhook Payload

```json
{
  "event": "order.shipped",
  "timestamp": "2024-01-22T16:45:00Z",
  "data": {
    "order_id": "ord_67890",
    "tracking_number": "1Z999AA10123456784",
    "carrier": "UPS"
  }
}
```

### Webhook Security

All webhook payloads include a signature in the `X-ACME-Signature` header. Verify this signature
using your webhook secret to ensure the request is authentic.

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The 'quantity' field must be a positive integer",
    "details": {
      "field": "quantity",
      "provided_value": -5
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_API_KEY | 401 | API key is missing or invalid |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INVALID_REQUEST | 400 | Request body is malformed |
| RESOURCE_NOT_FOUND | 404 | Requested resource doesn't exist |
| INSUFFICIENT_STOCK | 400 | Not enough inventory |
| INTERNAL_ERROR | 500 | Server error, contact support |

## SDKs and Libraries

Official SDKs are available for:
- Python: `pip install acme-api`
- JavaScript/Node.js: `npm install @acme/api-client`
- Ruby: `gem install acme_api`
- Go: `go get github.com/acmecorp/acme-go`

## Support

- Documentation: https://docs.acmecorp.com
- Developer Forum: https://community.acmecorp.com
- Email Support: api-support@acmecorp.com
- Status Page: https://status.acmecorp.com
