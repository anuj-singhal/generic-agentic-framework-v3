# RAG Test Conversations

Use these example conversations to test the Knowledge Base functionality. Each section shows queries you can ask after uploading the corresponding document.

---

## Test 1: Company Policy Document (`company_policy.txt`)

### Simple Questions

**Query 1:** "When was ACME Corporation founded and by whom?"
> Expected: Founded in 1985 by John Smith and Mary Johnson in Silicon Valley

**Query 2:** "How many days of PTO do new employees receive?"
> Expected: 15 days per year

**Query 3:** "What is the company's address?"
> Expected: 1234 Innovation Drive, San Jose, California 95134

---

### Benefits Questions

**Query 4:** "What percentage of health insurance premiums does the company pay?"
> Expected: 80%

**Query 5:** "How much is the 401k company match?"
> Expected: 4% company match

**Query 6:** "What is the tuition reimbursement limit?"
> Expected: Up to $5,000 per year

**Query 7:** "Tell me about the gym membership benefit"
> Expected: $50/month gym membership subsidy

---

### Policy Questions

**Query 8:** "What are the password requirements for IT systems?"
> Expected: Minimum 12 characters, must include uppercase, lowercase, numbers, special characters, changed every 90 days, cannot reuse last 10 passwords

**Query 9:** "How many days can employees work remotely?"
> Expected: Up to 3 days per week with manager approval

**Query 10:** "What is the expense limit for meals when traveling domestically?"
> Expected: Up to $75/day domestic

**Query 11:** "Within how many days must expenses be submitted?"
> Expected: Within 30 days of incurring them

---

### HR Questions

**Query 12:** "When are performance reviews conducted?"
> Expected: Annual reviews in Q1, mid-year check-ins in July

**Query 13:** "What is the typical merit increase range for employees who meet expectations?"
> Expected: 2-5% for "Meets Expectations"

**Query 14:** "What holidays does the company observe?"
> Expected: New Year's Day, MLK Day, Presidents' Day, Memorial Day, Independence Day, Labor Day, Thanksgiving + day after, Christmas Eve + Christmas Day

---

## Test 2: API Documentation (`api_documentation.md`)

### Authentication Questions

**Query 15:** "How do I authenticate with the ACME API?"
> Expected: Use API key in X-API-Key header, get key from Developer Portal at developers.acmecorp.com

**Query 16:** "What is the base URL for the API?"
> Expected: https://api.acmecorp.com/v2

---

### Rate Limits Questions

**Query 17:** "What are the rate limits for different plans?"
> Expected: Free: 60/min, 1000/day; Basic: 300/min, 10000/day; Pro: 1000/min, 100000/day; Enterprise: Unlimited

**Query 18:** "What headers show my remaining API rate limit?"
> Expected: X-RateLimit-Remaining header

---

### Endpoint Questions

**Query 19:** "How do I get a list of products from the API?"
> Expected: GET /products with optional query params (page, limit, category, min_price, max_price, in_stock)

**Query 20:** "What endpoint do I use to create an order?"
> Expected: POST /orders with customer_email, shipping_address, items, payment_method_id

**Query 21:** "What are the possible order statuses?"
> Expected: pending, confirmed, shipped, delivered, cancelled, refunded

**Query 22:** "How do I check inventory for a product?"
> Expected: GET /inventory/{product_id}

---

### Webhook Questions

**Query 23:** "What webhook events are available?"
> Expected: order.created, order.confirmed, order.shipped, order.delivered, order.cancelled, inventory.low

**Query 24:** "How do I verify webhook authenticity?"
> Expected: Verify X-ACME-Signature header using webhook secret

---

### Error Handling Questions

**Query 25:** "What error code is returned for rate limit exceeded?"
> Expected: RATE_LIMIT_EXCEEDED with HTTP 429

**Query 26:** "What SDK libraries are available?"
> Expected: Python (pip install acme-api), JavaScript (npm install @acme/api-client), Ruby (gem install acme_api), Go (go get github.com/acmecorp/acme-go)

---

## Test 3: Product Catalog (`product_catalog.csv`)

### Price Questions

**Query 27:** "What is the price of the Super Widget Pro?"
> Expected: $149.99

**Query 28:** "What is the most expensive product in the catalog?"
> Expected: Industrial Gadget X1 at $499.99

**Query 29:** "What is the cheapest product?"
> Expected: Connector Cable 2m at $14.99

---

### Product Details

**Query 30:** "What products are in the Safety category?"
> Expected: Safety Helmet Pro ($129.99), Safety Goggles Elite ($34.99), High-Vis Vest ($24.99)

**Query 31:** "What is the description of the Industrial Gadget X1?"
> Expected: Heavy-duty gadget for industrial applications with IP68 rating

**Query 32:** "Which products have the largest battery capacity?"
> Expected: Battery Pack Extended with 15000mAh

---

### Supplier Questions

**Query 33:** "What products are supplied by TechSupply Inc?"
> Expected: Super Widget Pro, Basic Widget, Portable Gadget Mini, Maintenance Kit, Wireless Adapter

**Query 34:** "Who supplies the safety equipment?"
> Expected: SafetyFirst Inc

---

### Specifications

**Query 35:** "What is the heaviest product?"
> Expected: Power Tool Bundle at 5.0 kg

**Query 36:** "How much stock is available for the Basic Widget?"
> Expected: 1000 units

---

## Test 4: FAQ Document (`faq.json`)

### Shipping Questions

**Query 37:** "How long does standard shipping take?"
> Expected: 5-7 business days within continental US

**Query 38:** "Do you ship internationally?"
> Expected: Yes, to over 50 countries, customers may be responsible for customs duties

**Query 39:** "Can I change my order after placing it?"
> Expected: Yes, within 2 hours of placement; after that, contact orders@acmecorp.com

---

### Returns & Warranty

**Query 40:** "What is the return policy?"
> Expected: 30-day money-back guarantee, items must be in original packaging and unused, refunds processed in 5-7 business days

**Query 41:** "How long is the standard warranty?"
> Expected: 2-year manufacturer warranty, extended options available (3-year, 5-year, lifetime)

**Query 42:** "How do I register my product for warranty?"
> Expected: Visit warranty.acmecorp.com, enter serial number and proof of purchase within 30 days

---

### Technical Support

**Query 43:** "How do I factory reset my device?"
> Expected: Power off, hold Reset button 10 seconds with paperclip, press Power while holding Reset, release when LED flashes blue

**Query 44:** "My device won't turn on, what should I do?"
> Expected: Charge for 30 minutes, try different cable/adapter, hard reset (hold power 15 seconds), contact support if persists

**Query 45:** "How do I update firmware?"
> Expected: Connect to WiFi, go to Settings > System > Software Update, download and install, keep connected to power

**Query 46:** "What temperature range should devices operate in?"
> Expected: 32°F to 95°F (0°C to 35°C), humidity below 80%

---

### Account & Payment

**Query 47:** "What payment methods are accepted?"
> Expected: Visa, MasterCard, Amex, Discover, PayPal, Apple Pay, Google Pay, ACME Gift Cards, Affirm financing for orders over $500

**Query 48:** "Do you offer business accounts?"
> Expected: Yes, with volume discounts (10% off $1000+, 15% off $5000+), NET-30 terms, dedicated account manager, priority support

**Query 49:** "What is the technical support phone number?"
> Expected: 1-800-ACME-TECH, available 24/7

---

## Test 5: Multi-Document Questions

After uploading ALL documents, try these cross-document queries:

**Query 50:** "What is the company's mission and what products do they sell?"
> Expected: Combines company_policy.txt (mission statement) with product_catalog.csv (product categories)

**Query 51:** "If I want to order products via API, what's the process and return policy?"
> Expected: Combines api_documentation.md (POST /orders) with faq.json (return policy)

**Query 52:** "What safety products does ACME sell and what warranty do they have?"
> Expected: Combines product_catalog.csv (Safety category products) with faq.json (2-year warranty)

---

## Verification Steps

1. **Without RAG:** Ask "What is ACME's return policy?" before uploading any documents
   - Should get generic response

2. **With RAG:** Upload `faq.json`, ingest it, then ask the same question
   - Should get specific answer: "30-day money-back guarantee..."

3. **Check Context Indicator:** Look for "📚 Knowledge Base Active" message in Chat tab

4. **Check Relevance:** Ask unrelated question like "What is the capital of France?"
   - RAG context should not interfere with general knowledge questions

---

## Tips for Testing

- Start with single document tests before multi-document
- Try rephrasing questions to test semantic understanding
- Ask follow-up questions to test context continuity
- Compare responses with/without documents loaded
- Check the ReAct trace to see if context is being used
