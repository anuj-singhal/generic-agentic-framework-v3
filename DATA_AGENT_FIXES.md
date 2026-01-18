# Data Agent Fixes and New Workflow

## Issues Fixed

### 1. Connection Errors
**Problem**: "Connection error" messages in medium and complex scenarios
**Root Cause**:
- Database connection not properly tested/reset on errors
- SQL queries with formatting issues (backticks, markdown code blocks)
- No connection health check

**Solution**:
- Added connection testing in `get_connection()` - runs `SELECT 1` to verify
- Automatic connection reset on errors
- Enhanced SQL cleaning in `execute_query()` to handle backticks, markdown, multiple statements

### 2. Tool Flow Confusion
**Problem**: Tools returned "instruction prompts" which confused the agent
**Root Cause**:
- Tools like `get_relevant_schema`, `generate_sql`, `validate_sql` returned long instruction strings
- Agent didn't know it needed to act on those instructions vs just passing them through
- No clear action steps

**Solution**:
- Completely restructured tools around **decomposition-based workflow**
- Tools now return concise, actionable guides
- Clear step-by-step process the agent must follow

## New Decomposition-Based Workflow

### Core Concept
Instead of trying to solve complex queries in one shot, the agent now:
1. **Decomposes** the query into smaller subtasks
2. **Executes** each subtask sequentially
3. **Synthesizes** the results into a final answer

### New Tools

#### 1. `decompose_query(user_query)` - FIRST STEP
**Purpose**: Break complex queries into manageable subtasks

**Returns**: Analysis guide with:
- Database schema with sample data (2 rows per table)
- Instructions to break query into ST1, ST2, ST3, etc.
- Each subtask includes: description, tables, dependencies, expected output

**When to use**: For MEDIUM and COMPLEX queries

#### 2. `get_schema_for_subtask(subtask_description)`
**Purpose**: Get detailed schema for a specific subtask

**Returns**:
- Full schema for all tables
- Sample data (3 rows per table)
- Relationships

**When to use**: Before generating SQL for each subtask

#### 3. `generate_sql_for_subtask(subtask_description, dependencies_results)`
**Purpose**: Guide for writing SQL for ONE subtask

**Returns**:
- Focus instructions for this specific subtask
- Schema quick reference
- How to handle dependencies from previous subtasks

**When to use**: For each subtask after getting schema

#### 4. `quick_validate_sql(sql_query)`
**Purpose**: Fast PASS/FAIL validation

**Returns**:
- "VALIDATION PASSED: Ready to execute" OR
- "VALIDATION FAILED: [specific error message]"

**When to use**: Before executing every SQL query

#### 5. `execute_sql(sql_query, max_rows=100)`
**Purpose**: Execute validated SQL

**Returns**: Query results as CSV format

**When to use**: After validation passes

#### 6. `synthesize_results(subtask_results, original_query)`
**Purpose**: Combine all subtask results into final answer

**Returns**: Guide for formatting final answer

**When to use**: After ALL subtasks are executed

### Workflow Example

**User Query**: "Calculate total portfolio value for each client and show the top 3"

**Step 1 - Decompose**:
```
Agent calls: decompose_query(query)
Agent analyzes and creates:
  ST1: Calculate portfolio values from HOLDINGS
  ST2: Join with CLIENTS to get names
  ST3: Rank and limit to top 3
```

**Step 2 - Execute Subtasks**:
```
For ST1:
  - get_schema_for_subtask("ST1")
  - Generate SQL: SELECT PORTFOLIO_ID, SUM(QUANTITY * AVG_COST) as VALUE FROM HOLDINGS GROUP BY PORTFOLIO_ID
  - quick_validate_sql(sql) → PASSED
  - execute_sql(sql) → Store ST1 results

For ST2:
  - Generate SQL: SELECT c.FULL_NAME, SUM(h.VALUE) FROM (ST1 concept) JOIN PORTFOLIOS p JOIN CLIENTS c ...
  - Actually writes full query using base tables
  - quick_validate_sql(sql) → PASSED
  - execute_sql(sql) → Store ST2 results

For ST3:
  - Could be combined with ST2 using ORDER BY + LIMIT
  - execute_sql(combined_sql) → Store ST3 results
```

**Step 3 - Synthesize**:
```
synthesize_results(
  "ST1: Portfolio values calculated for 15 portfolios
   ST2: Client totals summed for 10 clients
   ST3: Top 3 clients identified",
  original_query
)
→ Agent formats final answer with top 3 clients
```

## Key Changes Summary

| Old Approach | New Approach |
|-------------|--------------|
| Single complex SQL attempt | Break into subtasks |
| Long instruction prompts | Short, actionable guides |
| LLM validates itself | Fast automated validation |
| Template-based | Decomposition-based |
| No clear steps | Step-by-step workflow |
| 10 max iterations | 15 max iterations |
| Confusing tool returns | Clear PASS/FAIL signals |

## Updated Tools (12 Total)

**Decomposition Workflow** (Primary):
1. `decompose_query` - Break query into subtasks ✨ NEW
2. `get_schema_for_subtask` - Get schema for subtask ✨ UPDATED
3. `generate_sql_for_subtask` - SQL generation guide ✨ UPDATED
4. `quick_validate_sql` - Fast validation ✨ NEW
5. `execute_sql` - Execute query ✅ ENHANCED
6. `synthesize_results` - Combine results ✨ NEW

**Helper Tools**:
7. `show_tables` - List all tables
8. `describe_table` - Table schema
9. `get_sample_data` - Sample rows
10. `summarize_table` - Statistics
11. `analyze_query_complexity` - Determine complexity
12. `explain_sql` - Explain SQL in English

## Testing

### Simple Query Test
```bash
python test_data_agent.py
```
Query: "Show me all clients in the database"
Expected: Direct SQL, no decomposition needed

### Medium Query Test
```bash
python run_tests.py --category wealth_medium
```
Example: "Show clients with their portfolios"
Expected: 2-3 subtasks, JOIN operations

### Complex Query Test
```bash
python run_tests.py --category wealth_complex
```
Example: "Portfolio valuation analysis with top 3 clients"
Expected: 3-5 subtasks, CTEs/aggregations, synthesis

## Benefits

### For Simple Queries
- Faster execution (no decomposition overhead)
- Direct SQL generation
- Quick validation

### For Medium/Complex Queries
- **Breaks down complexity** into manageable pieces
- **Each subtask is simple** - agent writes better SQL
- **Better error handling** - validation catches issues early
- **Clearer reasoning** - can see the step-by-step process
- **More reliable** - less likely to fail on complex queries

### For Debugging
- Can see exactly which subtask failed
- Clear validation messages
- Step-by-step execution trace
- Better error messages

## Migration Notes

### Old Tools (Removed/Updated)
- `get_relevant_schema` → `get_schema_for_subtask` (more focused)
- `generate_sql` → `generate_sql_for_subtask` (subtask-specific)
- `validate_sql` → `quick_validate_sql` (simpler, faster)

### New Agent Behavior
- **Max iterations increased**: 10 → 15 (for subtask execution)
- **System prompt completely rewritten**: Decomposition-first approach
- **Tool count increased**: 10 → 12 tools

## Common Patterns

### Pattern 1: Simple Single-Table Query
```
1. Write SQL directly
2. quick_validate_sql(sql)
3. execute_sql(sql)
4. Return results
```

### Pattern 2: Medium JOIN Query
```
1. decompose_query → 2-3 subtasks
2. For each subtask:
   - Generate SQL
   - Validate
   - Execute
3. synthesize_results
```

### Pattern 3: Complex Multi-Step Analysis
```
1. decompose_query → 3-5 subtasks
2. Execute independent subtasks in parallel (if any)
3. Execute dependent subtasks sequentially
4. synthesize_results with formatting
```

## Troubleshooting

### "Connection error"
- Check if `agent_ddb.db` exists
- Run `python init_duckdb.py` to recreate
- Database is automatically tested on connection

### "VALIDATION FAILED"
- Check exact error message
- Common issues: typos in column names, missing JOINs, syntax errors
- Fix SQL and validate again

### "Empty results"
- Check if query filters are too restrictive
- Use `get_sample_data` to see what's actually in tables
- Verify JOIN conditions are correct

### Too many iterations
- Query might be too complex
- Try breaking into more subtasks manually
- Check if subtasks have circular dependencies

## Next Steps

To use the new workflow:

1. **Install/Update**:
   ```bash
   pip install -r requirements.txt
   python init_duckdb.py
   ```

2. **Test Simple Query**:
   ```bash
   python test_data_agent.py
   ```

3. **Test Medium Query**:
   ```bash
   python run_tests.py --category wealth_medium
   ```

4. **Test Complex Query**:
   ```bash
   python run_tests.py --category wealth_complex
   ```

5. **Use in Streamlit**:
   ```bash
   streamlit run app.py
   # Select "data_agent" from sidebar
   ```

## Expected Improvements

- ✅ No more connection errors (enhanced error handling)
- ✅ Better success rate on complex queries (decomposition)
- ✅ Clearer error messages (fast validation)
- ✅ More reliable SQL generation (one subtask at a time)
- ✅ Better final answers (synthesis step)

---

**Status**: Ready for testing
**Last Updated**: 2026-01-18
