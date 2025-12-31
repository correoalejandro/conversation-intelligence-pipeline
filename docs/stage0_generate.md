# Stage 0 — Ingest / Generate  
## Script documentation: `_logic_layer/0.addSyntheticConversations.py`

### Purpose
Stage 0 is the entry point of the pipeline. It generates or ingests **structured, analysis-ready conversational data** in batch form, establishing full metadata lineage for downstream embedding, clustering, labeling, and reporting stages.

---

### What the script does

1. **Prompt-driven specification**
   - Reads a markdown prompt selected at runtime.
   - Prompts act as domain specifications: conversational rules, tone, constraints, and a catalog of scenarios.
   - New prompts can be added or modified without changing code.

2. **Scenario-controlled generation**
   - Each conversation is generated under exactly one scenario.
   - Scenarios control intent, tone, and progression.
   - Scenario selection can be fixed or randomized per batch.

3. **Controlled synthetic conversation generation**
   - Uses an LLM as a data generator, not a chatbot.
   - Enforces:
     - Agent ↔ Client alternation
     - Fixed conversation length (prompt-dependent)
     - Linguistic constraints (Spanish call-center style)

4. **Stateful interaction modeling**
   - Maintains conversational coherence across turns.
   - Ensures scenario consistency and behavioral progression.

5. **Temporal modeling (optional, prompt-dependent)**
   - Emits ISO-8601 timestamps per turn.
   - Applies role-dependent response latency.
   - Produces chronologically consistent interaction records.

6. **Batch-oriented execution**
   - Operates on batches as first-class units.
   - Creates deterministic folder and file structures per batch.
   - Generates batch artefacts:
     - JSON (metadata + conversations)
     - TXT preview for human inspection

7. **Interactive, operator-driven execution**
   - Menu-based interface (no CLI flags required).
   - Operator selects:
     - Prompt
     - Scenario mode
     - Total conversation target / batch size
   - Human input configures behavior; execution is automated.

8. **Data-layer integration and lineage**
   - Registers all artefacts using the data layer:
     - `register_prompt(...)` → stores prompt metadata, returns `prompt_id`
     - `register_batch(...)` → stores batch metadata, returns `batch_id`
     - `register_conversations(...)` → stores conversation-level metadata
     - `audit_batch(batch_id)` → validates metadata consistency
   - Enables downstream retrieval by metadata rather than filesystem scans.

9. **Concurrency safety**
   - Uses file-level locking to prevent batch artefact corruption.
   - Supports safe parallel or repeated runs.

10. **Downstream readiness**
    - Outputs are immediately consumable by:
      - Embedding pipelines
      - Clustering and batch analysis
      - Labeling workflows
      - Dashboards and BI tools (e.g., SAS Viya)

---

### Inputs
- Markdown prompt file (selected interactively)
- Operator configuration (menu-driven)
- Optional Azure OpenAI credentials (falls back to mock generation if absent)

---

### Outputs
- Batch folder with:
  - Individual conversation JSON files
  - Batch-level JSON artefact
  - Batch-level TXT preview
- Registry records for prompts, batches, and conversations

---

### Mental model
Stage 0 is a **prompt-driven, operator-configured engine** that produces **structured conversational data in batch form**, forming the foundation for all subsequent analytical stages.
