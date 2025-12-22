# Conversation Intelligence Pipeline

Conversation Intelligence Pipeline is an end-to-end AI pipeline designed to support the operational analysis of conversational data in customer service and debt-collection workflows. It helps teams transform raw conversations into structured semantic insights that can be explored, labeled, and connected to business actions such as follow-up, escalation, and customer segmentation.

The pipeline was built for applied, real-world use: it supports both production-style data ingestion and controlled synthetic data generation, allowing teams to prototype, test, and demonstrate conversational analytics workflows even when real customer data cannot be shared.

## Why this pipeline exists

Organizations handling large volumes of customer conversations face two recurring problems:
- Manual review does not scale.
- Patterns related to intent, risk, or escalation are difficult to surface consistently.

This pipeline addresses those problems by:
- Structuring conversations using embeddings and clustering.
- Extracting behavioral signals that can guide operational decisions.
- Producing outputs that are interpretable and exportable to reporting systems.

Rather than replacing human agents or analysts, the system supports them by organizing conversational data into actionable, reviewable structures.

## Key design features

- **Controlled AI-based synthetic conversation generation**  
  The pipeline can generate realistic Spanish call-center conversations in a controlled manner, enabling safe experimentation, scenario testing, and demonstrations without exposing sensitive data.

- **Pipeline design from ingestion to export**  
  The system is structured as a multi-stage pipeline: ingest/generate → embed → cluster → normalize → label → export, making it easier to reason about, debug, and extend.

- **Analysis at multiple levels**  
  Supports both conversation-level and message-level embeddings and clustering, allowing analysis at different granularities depending on the operational need.

- **Designed for downstream decision workflows**  
  Outputs are structured for dashboards, reporting, and BI tools (SAS CAS), not only for exploratory research.

## What the pipeline does (high-level)

- Ingests or synthesizes conversation batches.
- Converts text into numeric representations using embeddings.
- Groups similar conversations or messages via clustering.
- Cleans and normalizes Spanish text.
- Extracts keywords per cluster to produce human-readable labels.
- Exports result tables to SAS CAS for reporting and analysis.

## Pipeline stages and entry points

### Stage 0 — Ingest / Generate

- `_logic_layer/0.v12.addSyntheticConversations.py`  
  Registers conversation batches from JSON/JSONL files so they can be used by the rest of the pipeline.

- `_logic_layer/0.v13.synthetic_cobranza_generator.py`  
  Generates synthetic Spanish call-center conversations in a controlled manner for demos and testing.

- `_logic_layer/0.v14.scenario_Control.py`  
  Runs predefined synthetic scenarios to exercise the pipeline end to end.

### Stage 1 — Embeddings

- `_logic_layer/1.v11.embedd_conversations.py`  
  Creates conversation-level embeddings, producing one numeric representation per conversation.

- `_logic_layer/1.v12.embed_messages_menu.py`  
  Creates message-level embeddings through a menu-driven workflow, producing one representation per message.

### Stage 2 — Clustering and batch analysis

- `_logic_layer/2.v13.embedding_analysis.py`  
  Clusters conversation-level embeddings to group similar interactions.

- `_logic_layer/2.v14.batch_selector.py`  
  Allows selection and inspection of batches before analysis.

- `_logic_layer/2.v15.batch_analysis_menu.py`  
  Clusters message-level embeddings via a menu-driven interface.

### Stage 3 — Text processing

- `_logic_layer/3.v13.text_process.py`  
  Cleans and normalizes Spanish text (e.g., standardization and preprocessing) for downstream steps.

### Stage 4 — Cluster labeling

- `_logic_layer/4.v11.topicExtractor.py`  
  Extracts top keywords per cluster and produces a first preview of cluster labels.

- `_logic_layer/4.v12.topicExtractor.py`  
  Produces similar outputs with improved vocabulary handling and a confirmation step.

- `_logic_layer/4.v13.topicExtractor.py`  
  Adds richer configuration options and optional AI-generated summaries for clusters.

- `_logic_layer/4.v14.topicExtractor.py`  
  Produces merged tables that join clusters with their extracted keywords.

### Stage 5 — Export

- `_logic_layer/5.v5.uploadToCAS.py`  
  Uploads result tables to SAS CAS so they can be used in reporting or BI tools.

## Project structure (reference)

- `_logic_layer/` — Core pipeline stages and logic.
- `_presentation_layer/` — Menu-based interfaces and UI scripts to run the pipeline.
- `_connections_layer/` — Connectors to external services or data sources.
- `_2_new_logic/` — Experimental or exploratory scripts and notebooks.
- `6.v7.web_ui_app.py` — Web UI entry point.

## Notes

This repository contains an applied pipeline under active development. Multiple versioned scripts coexist to support experimentation and comparison across different pipeline variants. For demos or operational runs, only a subset of these versions is typically used.
