# NYC Real Estate GPT (LangChain RAG) – Round 1 Interview/Screening

Welcome to the **NYC Real Estate GPT** project! This repository demonstrates a basic **Retrieval-Augmented Generation (RAG)** pipeline utilizing data from the [NYCDB repository](https://github.com/nycdb/nycdb), specifically the **ACRIS** (Automated City Register Information System) dataset. Our goal is to create a "ChatGPT for NYC Real Estate," providing actionable insights into property details, legal information, permits, and more.

> **Assignment Due Date**: Friday, January 10, 2025

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [How It Works](#how-it-works)
6. [Demonstration & Sample Queries](#demonstration--sample-queries)
7. [Proof of Concept](#proof-of-concept)
8. [Future Improvements](#future-improvements)

---

## Project Overview

In real estate development and investment, stakeholders require quick access to vast amounts of property-related data (e.g., legal details, ownership, size, year built, etc.). This project aims to showcase how an LLM (Large Language Model) can be integrated with a vector database (ChromaDB) to retrieve relevant documents from a large dataset (ACRIS) and provide a concise, human-readable answer. 

**Key Components**:
- **Data Processing**: Download and preprocess ACRIS data.
- **Vector Store**: Use embeddings (e.g., `text-embedding-ada-002`) to store document representations in ChromaDB.
- **Retrieval**: Given a user query, retrieve top-k relevant documents.
- **Generation**: Generate a final answer using an LLM, augmented by the retrieved documents.

---

## Dataset

We leverage the **ACRIS** data from the [NYCDB repository](https://github.com/nycdb/nycdb). The ACRIS dataset contains:

- **Property Legal Info**: Ownership, legal documents, and records.
- **Property Details**: Size, year built, address, etc.

Please refer to the [NYCDB Wiki: ACRIS Dataset](https://github.com/nycdb/nycdb/wiki) for instructions on how to obtain and structure this data. After downloading, ensure you have the relevant files (e.g., CSV/JSON) in the `data/` directory for processing.

---

## Project Structure

Below is an overview of the main folders and files in this repository:

```
.
├── .chromadb/                  # ChromaDB persistence directory (created at runtime)
├── ollama/
│   └── Dockerfile             # Docker setup for Ollama if needed
├── streamlit_app/
│   ├── data/
│   │   ├── acris_real_property_legals.csv
│   │   ├── acris_real_property_legals_processed.json
│   │   └── processed_acris_data.csv
│   ├── llms/
│   │   ├── __init__.py
│   │   ├── clients.py         # LLM client classes for OpenAI & Ollama
│   │   └── config.py          # Manages configuration (e.g., environment variables)
│   ├── data_ingestor.py       # Processes and inserts data into ChromaDB
│   ├── app.py                 # Streamlit app for the RAG chatbot
│   └── ...
├── docker-compose.yml
├── pyproject.toml
├── poetry.lock
├── README.md                  # You are here!
└── ...
```

**Key Files**:
- **`clients.py`**: Defines `OpenAILLMSClient` and `OllamaLLMSClient`. Each client handles:
  - **LLM chat completions** (e.g., GPT-4, Llama-based models).
  - **Embeddings** (e.g., `text-embedding-ada-002`).
- **`app.py`**: Streamlit application that:
  - Displays a chat interface.
  - Retrieves relevant documents from ChromaDB.
  - Calls an LLM with both the user question and retrieved context.
- **`data_ingestor.py`**:
  - Reads the preprocessed JSON file.
  - Creates or retrieves a ChromaDB collection.
  - Embeds and upserts documents into the vector store.
  - Provides a retrieval method for top-k relevant documents.
- **`config.py`**: Manages environment variables and settings such as API keys and base URLs.

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/NIKHILKUMARREDDY28/NYC-Real-Estate-GPT
   ```

2. **Install Dependencies**  
   We recommend using [Poetry](https://python-poetry.org/docs/) for dependency management:
   ```bash
   poetry install
   ```

   Alternatively, you can use a virtual environment and install requirements manually (not shown here).

3. **Obtain the ACRIS Dataset**  
   - Visit the [NYCDB repository](https://github.com/nycdb/nycdb) and follow the instructions for ACRIS data.
   - Download the dataset (CSV or JSON) and place the files under `streamlit_app/data/`.
   - *Example*: `acris_real_property_legals_processed.json` with embedded text fields.

4. **Configure Environment Variables**  
   - In `streamlit_app/.env` (or wherever your config points), set:
     ```bash
     OPENAI_API_KEY=YOUR_OPENAI_API_KEY
     OLLAMA_API_URL=http://localhost:11434/v1 # Example for local Ollama
     ```
   - Check `streamlit_app/config.py` to confirm how these variables are loaded.

5. **Ingest Data into ChromaDB**  
 
   This step reads the JSON/CSV, computes embeddings for each document, and inserts them into a ChromaDB collection.

6. **Launch the Streamlit App**  
   ```bash
   streamlit run app.py
   ```
   Open the provided URL (e.g., `http://localhost:8501`) in your browser to interact with the chatbot.

---

## How It Works

1. **User Query**  
   The user enters a query such as *“What permits are required for renovations at this property?”* in the Streamlit UI.

2. **Embedding & Retrieval**  
   - The user’s query is embedded using `text-embedding-ada-002`.
   - The embedded query is compared against the stored embeddings in ChromaDB.
   - The top *k* relevant documents are retrieved (e.g., property details, legal documents).

3. **Augmented Generation**  
   - The retrieved documents are injected into the LLM’s prompt (as context).
   - The LLM (GPT-4 or Ollama-based model) then produces a comprehensive answer using both the user query and the contextual documents.

4. **Response**  
   - The response is displayed in the Streamlit chat interface.
   - The conversation history and system messages are maintained in `st.session_state`.

---

## Demonstration & Sample Queries

After launching the app, try the following example queries:

1. **“Who owns this address?”**  
   - The system retrieves relevant ownership records from ACRIS.
   - The LLM responds with the current property owner (if available in the data).

2. **“What permits are required for renovations at 123 Main Street?”**  
   - Retrieves legal documents indicating if certain building permits were recorded in ACRIS.
   - Provides guidance and references based on the retrieved context.

3. **“How many times has this property changed ownership since 2010?”**  
   - Summarizes the relevant ACRIS documents that indicate ownership transfers.

As each query is processed, the Streamlit interface will display the user’s request, the relevant context documents, and the final answer from the LLM.

---

## Proof of Concept

1. **Code & Pipeline**  
   - [**`data_ingestor.py`**](./streamlit_app/data_ingestor.py): Demonstrates how data is ingested, embedded, and stored in ChromaDB.  
   - [**`clients.py`**](./streamlit_app/llms/clients.py): Showcases how both OpenAI and Ollama are integrated to provide LLM-based responses.

2. **Performance & Logging**  
   - You can insert print/log statements to see how many documents are retrieved, the similarity scores, and the final answers.
   - ChromaDB allows you to adjust the similarity search parameters (e.g., `k` and distance thresholds).

3. **Testing**  
   - Local tests can be done by running the Streamlit app and verifying the returned answers.
   - Additional validation can be performed by adjusting the number of retrieved documents and observing changes in the answer quality.

---

## Future Improvements

1. **Additional Datasets**  
   - Integrate other NYC datasets (e.g., **PLUTO**, **DOB** permits) for more comprehensive coverage.

2. **Advanced Query Parsing**  
   - Implement more sophisticated query parsing or question answering logic (e.g., extracting addresses or property identifiers from the query).

3. **Model Fine-Tuning**  
   - Perform fine-tuning or instruction tuning on the LLM specifically for real estate/legal documents.

4. **UI Enhancements**  
   - Enhance the Streamlit interface with filter options (e.g., date ranges, borough selections).
   - Add visualization components such as maps or charts for property data exploration.

5. **Scalability**  
   - Explore distributed or cloud-hosted vector databases for larger-scale data.
   - Integrate with external APIs for real-time data (e.g., building violation lookup).

---
