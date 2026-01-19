# RAG Test Documents

This folder contains sample documents for testing the Knowledge Base (RAG) functionality.

## Documents Included

| File | Type | Description |
|------|------|-------------|
| `company_policy.txt` | TXT | Employee handbook with HR policies, benefits, expenses |
| `api_documentation.md` | MD | Technical API documentation with endpoints, examples |
| `product_catalog.csv` | CSV | Product catalog with 20 items, prices, specifications |
| `faq.json` | JSON | Customer FAQ with 17 Q&A pairs across 4 categories |

## How to Test

1. Start the application: `streamlit run app.py`
2. Enter your OpenAI API key in the sidebar
3. Go to the **"📚 Knowledge Base"** tab
4. Upload one or more documents from this folder
5. Click **"Ingest"** for each document
6. Go to the **"💬 Chat"** tab
7. Ask questions related to the documents

## Sample Test Queries

### Company Policy Questions
- "How many PTO days do employees get after 5 years?"
- "What is the expense limit for hotels when traveling internationally?"
- "What are the password requirements at ACME?"
- "How often are performance reviews conducted?"
- "What percentage of health insurance premiums does the company pay?"
- "Who founded ACME Corporation and when?"
- "What is the gym membership subsidy?"

### API Documentation Questions
- "How do I authenticate with the ACME API?"
- "What are the rate limits for the Pro plan?"
- "How do I create an order using the API?"
- "What webhook events are available?"
- "What error code is returned for invalid API key?"
- "What SDKs are available for the ACME API?"
- "How do I check product stock via the API?"

### Product Catalog Questions
- "What is the price of the Super Widget Pro?"
- "Which products are in the Safety category?"
- "What is the most expensive product in the catalog?"
- "How many products are supplied by TechSupply Inc?"
- "What is the battery capacity of the Extended Battery Pack?"
- "Which product has a weight of 5kg?"

### FAQ Questions
- "What is ACME's return policy?"
- "How long does international shipping take?"
- "How do I reset my device to factory settings?"
- "What payment methods does ACME accept?"
- "How do I register my product for warranty?"
- "What is the difference between Widget Pro and Widget Deluxe?"
- "What are the recommended operating temperatures?"

## Expected Behavior

When you ask these questions:
1. The system automatically searches the knowledge base
2. Relevant document chunks are found using semantic similarity
3. Context is injected into the prompt (you'll see "Knowledge Base Active" indicator)
4. The AI agent uses this context to provide accurate answers

## Verification

To verify RAG is working:
1. Ask a question WITHOUT documents loaded - get generic response
2. Upload and ingest the relevant document
3. Ask the same question - get specific, accurate response based on document content

## Notes

- Documents are stored in memory only (cleared on app restart)
- Each document is split into ~1000 character chunks with 200 char overlap
- Top 5 most relevant chunks are included in context
- Minimum relevance score is 0.3 (30%)
