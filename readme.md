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

- **Controlled AI-based synthetic conversation generation** The pipeline can generate realistic Spanish call-center conversations in a controlled manner, enabling safe experimentation, scenario testing, and demonstrations without exposing sensitive data.

- **Pipeline design from ingestion to export** The system is structured as a multi-stage pipeline: ingest/generate → embed → cluster → normalize → label → export, making it easier to reason about, debug, and extend.

- **Analysis at multiple levels** Supports both conversation-level and message-level embeddings and clustering, allowing analysis at different granularities depending on the operational need.

- **Designed for downstream decision workflows** Outputs are structured for dashboards, reporting, and BI tools (SAS CAS), not only for exploratory research.

---

## 🛠 What the pipeline does (high-level)

- Ingests or synthesizes conversation batches.
- Converts text into numeric representations using embeddings.
- Groups similar conversations or messages via clustering.
- Cleans and normalizes Spanish text.
- Extracts keywords per cluster to produce human-readable labels.
- Exports result tables to SAS CAS for reporting and analysis.

---

## ⛓ Pipeline stages and entry points

### Stage 0 — Ingest / Generate
- `_logic_layer/0.v12.addSyntheticConversations.py`  
  Registers conversation batches from JSON/JSONL files.
- `_logic_layer/0.v13.synthetic_cobranza_generator.py`  
  Generates synthetic Spanish call-center conversations in a controlled manner.
- `_logic_layer/0.v14.scenario_Control.py`  
  Runs predefined synthetic scenarios for testing and demonstrations.

### Stage 1 — Embeddings
- `_logic_layer/1.v11.embedd_conversations.py`  
  Creates conversation-level embeddings.
- `_logic_layer/1.v12.embed_messages_menu.py`  
  Creates message-level embeddings via a menu-driven workflow.

### Stage 2 — Clustering and batch analysis
- `_logic_layer/2.v13.embedding_analysis.py`  
  Clusters conversation-level embeddings.
- `_logic_layer/2.v14.batch_selector.py`  
  Allows batch selection prior to analysis.
- `_logic_layer/2.v15.batch_analysis_menu.py`  
  Clusters message-level embeddings.

### Stage 3 — Text processing
- `_logic_layer/3.v13.text_process.py`  
  Cleans and normalizes Spanish text.

### Stage 4 — Cluster labeling
- `_logic_layer/4.v11.topicExtractor.py`  
  Extracts top keywords per cluster.
- `_logic_layer/4.v12.topicExtractor.py`  
  Improves vocabulary handling and adds confirmation.
- `_logic_layer/4.v13.topicExtractor.py`  
  Adds richer configuration and optional AI-generated summaries.
- `_logic_layer/4.v14.topicExtractor.py`  
  Produces merged tables joining clusters and keywords.

### Stage 5 — Export
- `_logic_layer/5.v5.uploadToCAS.py`  
  Uploads result tables to SAS CAS.

---

## 📂 Project structure (reference)

- `_logic_layer/` — Core pipeline stages.
- `_presentation_layer/` — Menu-based interfaces and UI scripts.
- `_connections_layer/` — External connectors.
- `_2_new_logic/` — Experimental scripts and notebooks.
- `6.v7.web_ui_app.py` — Web UI entry point.

---

## 📈 Repository Insights

<div align="center">
  <img height="165em" src="https://github-readme-stats.vercel.app/api?username=YOUR_USERNAME&show_icons=true&theme=tokyonight&count_private=true&hide_border=true"/>
  <img height="165em" src="https://github-readme-stats.vercel.app/api/top-langs/?username=YOUR_USERNAME&layout=compact&langs_count=8&theme=tokyonight&hide_border=true"/>
</div>

---

## 📝 Notes

This repository reflects an applied, evolving pipeline. Multiple versioned scripts coexist to support experimentation and comparison across different pipeline variants. For demos or operational runs, only selected versions are typically executed.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:005AA7,100:007BFF&height=100&section=footer"/>
</p>
