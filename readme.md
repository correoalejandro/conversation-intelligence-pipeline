<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:003973,100:005AA7&height=220&section=header&text=Conversation%20Intelligence%20Pipeline&fontSize=32&fontColor=ffffff&animation=fadeIn&fontAlignY=40"/>
</p>

<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=900&color=007BFF&center=true&vCenter=true&width=720&lines=AI-Powered+Conversational+Analytics+Pipeline;Semantic+Insights+|+Clustering+|+NLP;Built+for+Operational+Analysis+and+SAS+CAS+Integration)](https://git.io/typing-svg)

</div>

# Conversation Intelligence Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NLP-Spanish-005AA7?style=for-the-badge&logo=natural-language-processing&logoColor=white"/>
  <img src="https://img.shields.io/badge/SAS%20CAS-003366?style=for-the-badge&logo=sas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/JSON-000000?style=for-the-badge&logo=json&logoColor=white"/>
</p>

---
## 📑 Index

1. [Key design features](#-key-design-features)
2. [Competitive and differentiating aspects](#-competitive-and-differentiating-aspects)
3. [How to navigate and use this repository](#how-to-navigate-and-use-this-repository)
4. [What the pipeline does (high-level)](#-what-the-pipeline-does-high-level)
5. [Example outputs](#-example-outputs)
   - [Semantic exploration](#semantic-exploration)
   - [Operational dashboard](#operational-dashboard)
   - [From language patterns to actions](#from-language-patterns-to-actions)
6. [Pipeline stages and entry points](#-pipeline-stages-and-entry-points)
   - [Stage 0 — Ingest / Generate](#stage-0--ingest--generate)
   - [Stage 1 — Embeddings](#stage-1--embeddings)
   - [Stage 2 — Clustering and batch analysis](#stage-2--clustering-and-batch-analysis)
   - [Stage 3 — Text processing](#stage-3--text-processing)
   - [Stage 4 — Cluster labeling](#stage-4--cluster-labeling)
   - [Stage 5 — Export](#stage-5--export)
   - [Stage 6 — Presentation & Interaction](#stage-6--presentation--interaction)
7. [Project structure](#-project-structure)
8. [Notes](#-notes)
---


This project is an end-to-end AI pipeline designed to support operational analysis of conversational data in customer service and debt-collection workflows. It helps teams transform raw conversations into structured semantic insights that can be explored, labeled, and connected to business actions such as follow-up, escalation, and customer segmentation.

The pipeline was built for applied, real-world use: it supports both production-style data ingestion and controlled synthetic data generation, allowing teams to prototype, test, and demonstrate conversational analytics workflows even when real customer data cannot be shared.

---

## 🧭 Key design features

Organizations handling large volumes of customer conversations face two recurring problems:
- Manual review does not scale.
- Patterns related to intent, risk, or escalation are difficult to surface consistently.

This pipeline addresses those problems by:
- Structuring conversations using embeddings and clustering.
- Extracting behavioral signals that can guide operational decisions.
- Producing outputs that are interpretable and exportable to reporting systems.

Rather than replacing human agents or analysts, the system supports them by organizing conversational data into actionable, reviewable structures.

---

## 🚀 Competitive and differentiating aspects

- **Controlled AI-based synthetic conversation generation** - **Controlled AI-based synthetic conversation generation**  
  The pipeline includes a production-style synthetic data generator that produces
  scenario-driven, timestamped Spanish call-center conversations in batch form,
  with full metadata lineage, auditability, and human-readable previews.
  This enables safe experimentation, reproducible demos, and pipeline testing
  without exposing sensitive customer data.

- **Pipeline design from ingestion to export** The system is structured as a multi-stage pipeline: ingest/generate → embed → cluster → normalize → label → export, making it easier to reason about, debug, and extend.

- **Analysis at multiple levels** Supports both conversation-level and message-level embeddings and clustering, allowing analysis at different granularities depending on the operational need.

- **Designed for downstream decision workflows** Outputs are structured for dashboards, reporting, and BI tools (SAS Viya), not only for exploratory research.

---

### How to navigate and use this repository

The repository is organized so that each pipeline stage can be read, executed, and reasoned about independently.

- Readers can start from the **pipeline stages below** to understand the end-to-end flow.
- In the `_logic_layer/`, each stage may be implemented across a small set of scripts that cover batch selection, main processing, or interactive execution, depending on how that step is typically run.
- Supporting components (data access, persistence, presentation) are kept separate so the core processing logic remains easy to follow and adapt.

This structure supports both individual experimentation and reuse by other practitioners exploring conversational analytics pipelines.


---


## 🛠 What the pipeline does (high-level)

- Ingests existing conversation batches or generates controlled synthetic batches
  with full provenance, suitable for downstream embedding, clustering, and analysis.
- Converts text into numeric representations using embeddings.
- Groups similar conversations or messages via clustering.
- Cleans and normalizes Spanish text.
- Extracts keywords per cluster to produce human-readable labels.
- Exports result tables to SAS Viya (CAS) for reporting and analysis.

## 📊 Example outputs

Below are representative outputs produced by the pipeline at different stages of analysis.  
They illustrate how conversational data moves from semantic structure to operational insight.

### Semantic exploration
*Semantic map of customer messages grouped by similarity.*

![Semantic map of customer messages](_presentation_layer/assets/semantic_map_1.png)
![Semantic map of customer messages](_presentation_layer/assets/semantic_map_2.png)

### Operational dashboard
*Conversation intent and scenarios summarized by agent and time period.*

![Operational dashboard](_presentation_layer/assets/operational_dashboard.png)

### From language patterns to actions
*Example of how clustered message patterns are translated into recommended operational actions.*

![Text to action mapping](_presentation_layer/assets/text_to_action_mapping.png)

<small>Extended analytical reports and additional views are available separately.</small>


## ⛓ Pipeline stages and entry points

### Stage 0 — Ingest / Generate
- `_logic_layer/0.addSyntheticConversations.py`  


### Stage 1 — Embeddings
- `_logic_layer/1.embed_conversations.py`  
  Creates conversation-level embeddings.
- `_logic_layer/1.embed_messages_menu.py`  
  Creates message-level embeddings via an interactive CLI workflow.

### Stage 2 — Clustering and batch analysis
- `_logic_layer/2.embedding_analysis.py`  
  Clusters embeddings (UMAP + HDBSCAN) and computes batch-level summaries
- `_logic_layer/2.batch_selector.py`  
  Allows batch selection prior to analysis.
- `_logic_layer/2.batch_analysis_menu.py`  
  Runs message-level clustering and inspection via CLI.

### Stage 3 — Text processing
- `_logic_layer/3.text_process.py`  
  Cleans and normalizes Spanish text.

### Stage 4 — Cluster labeling
- `_logic_layer/4.topic_extractor.py`  
  Extracts and refines human-readable cluster labels (iterative improvements consolidated).

### Stage 5 — Export
- `_logic_layer/5.uploadToCAS.py`  
  Uploads result tables to SAS Viya (CAS).

### Stage 6 — Presentation & Interaction
- `_presentation_layer/6.cli_menu.py`  
  CLI-based launcher for running and inspecting pipeline stages.
- `_presentation_layer/6.web_ui.py`  
  Streamlit-based web UI for exploring results and summaries.
- `_presentation_layer/reporting_tools/`  
  Reporting and presentation artifacts (non-core, exploratory outputs).

---

## 📂 Project structure

- `_data_layer/` — Data models, paths, registry, and backends (I/O and artifact management).
- `_logic_layer/` — Core multi-stage pipeline logic (Stages 0–5).
- `_presentation_layer/` — Presentation and interaction layer (Stage 6).
- `_connections_layer/` — External connectors and integrations.

---

## 📝 Notes

This repository reflects an **applied, curated pipeline** developed through iterative experimentation.

Earlier exploratory variants and reporting artifacts have been **consolidated or archived** to keep the main execution paths clear and reviewable. The current structure highlights the **canonical pipeline stages (Stages 0–6)** used for demonstrations, analysis, and operational-style runs.
