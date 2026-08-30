# 🛡️ SBI Insurance Policy RAG Assistant

### Production-Oriented Retrieval-Augmented Generation Application for Insurance Policy Question Answering

A production-oriented **Retrieval-Augmented Generation (RAG)** application that enables users to ask natural-language questions about insurance policy documents and receive **context-grounded answers with source information**.

The application is built using **Python, LangChain, OpenAI, ChromaDB, and Streamlit** and includes production-focused capabilities such as **query rewriting, semantic retrieval, LLM-based reranking, relevance validation, grounding validation, groundedness validation, answer retry mechanisms, source attribution, and conversational memory**.

---

# 📌 Table of Contents

* [Overview](#-overview)
* [Business Problem](#-business-problem)
* [Solution](#-solution)
* [Key Features](#-key-features)
* [Architecture](#-architecture)
* [End-to-End RAG Workflow](#-end-to-end-rag-workflow)
* [Project Structure](#-project-structure)
* [Document Ingestion Pipeline](#-document-ingestion-pipeline)
* [Indexing Pipeline](#-indexing-pipeline)
* [Query Processing Pipeline](#-query-processing-pipeline)
* [Retrieval and Reranking](#-retrieval-and-reranking)
* [RAG Guardrails](#-rag-guardrails)
* [Grounded Answer Generation](#-grounded-answer-generation)
* [Conversation Management](#-conversation-management)
* [Streamlit Application](#-streamlit-application)
* [Application Screenshots](#-application-screenshots)
* [Testing](#-testing)
* [Project Results](#-project-results)
* [Technology Stack](#-technology-stack)
* [Installation](#-installation)
* [Environment Variables](#-environment-variables)
* [Running the Application](#-running-the-application)
* [Production Considerations](#-production-considerations)
* [Future Enhancements](#-future-enhancements)
* [Learning Outcomes](#-learning-outcomes)
* [Author](#-author)

---

# 📌 Overview

Insurance policy documents contain a large amount of structured and unstructured information such as:

* Policy benefits
* Coverage details
* Exclusions
* Eligibility criteria
* Claim conditions
* Medical expenses
* Policy limits
* Terms and conditions
* Waiting periods
* Policy-specific clauses

Finding the correct information manually from large policy documents can be time-consuming.

This project addresses this problem by implementing a **Retrieval-Augmented Generation architecture** that retrieves relevant policy information and provides answers using the retrieved context instead of relying only on the language model's internal knowledge.

The system is designed to minimize unsupported responses and improve answer reliability through multiple validation and guardrail mechanisms.

---

# 🎯 Business Problem

Insurance customers and employees may need to quickly find answers to questions such as:

> What medical expenses are covered under the policy?

> What are the exclusions under the policy?

> What conditions are covered?

> What are the eligibility requirements?

Searching manually through large PDF policy documents can be inefficient.

A traditional LLM chatbot also has a major limitation:

```text
User Question
      ↓
     LLM
      ↓
Generated Answer
```

The LLM may generate information that is not actually present in the policy document.

This project solves the problem using:

```text
User Question
      ↓
Retrieve Relevant Policy Information
      ↓
Build Grounded Context
      ↓
LLM
      ↓
Validate Answer
      ↓
Final Answer + Sources
```

---

# 💡 Solution

The application implements a complete RAG pipeline:

```text
Insurance Policy PDFs
        ↓
Document Ingestion
        ↓
Text Extraction
        ↓
Chunking
        ↓
OpenAI Embeddings
        ↓
ChromaDB
        ↓
Query Classification
        ↓
Query Rewriting
        ↓
Semantic Retrieval
        ↓
LLM Reranking
        ↓
Relevance Validation
        ↓
Context Construction
        ↓
Prompt Construction
        ↓
OpenAI LLM
        ↓
Grounding Validation
        ↓
Groundedness Validation
        ↓
Retry if Required
        ↓
Final Answer + Sources
```

---

# ✨ Key Features

## 📄 Document Processing

* PDF document ingestion
* Text extraction
* Document chunking
* Metadata preservation
* Policy-level document identification

## 🔎 Retrieval

* Semantic similarity search
* ChromaDB vector database
* Policy-based filtering
* Top-K document retrieval

## 🧠 Query Processing

* Query classification
* Query rewriting
* Improved retrieval queries

## 🎯 Reranking

* LLM-based reranking
* Ranking retrieved chunks based on query relevance
* Selection of the most useful context

## 🛡️ Guardrails

The application contains multiple validation mechanisms:

* Relevance Guard
* Grounding Guard
* Groundedness Guard
* Answer Retry Guard

## 📚 Source Attribution

The application tracks metadata associated with retrieved documents and provides source information with the generated answer.

## 💬 Conversation Management

The application maintains conversation history to support follow-up questions and multi-turn interactions.

## 🖥️ Streamlit UI

A simple interactive Streamlit interface allows users to:

* Select/query policy information
* Enter natural-language questions
* View generated answers
* View source information
* Ask follow-up questions

---

# 🏗️ Architecture

The application follows a modular production-oriented RAG architecture.

## High-Level Architecture

```text
                         ┌───────────────────────┐
                         │      Streamlit UI     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      RAG Service      │
                         └───────────┬───────────┘
                                     │
                       ┌─────────────┴─────────────┐
                       │                           │
                       ▼                           ▼
              Query Classification         Query Rewriting
                       │                           │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Semantic Retriever    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       ChromaDB        │
                         │    Vector Database    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    LLM Reranker       │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Relevance Guard    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Context Builder    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Prompt Builder     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      OpenAI LLM       │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌────────────────────┐          ┌─────────────────────┐
          │  Grounding Guard  │          │ Groundedness Guard  │
          └──────────┬─────────┘          └──────────┬──────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │   Answer Retry Guard  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Answer + Sources      │
                         └───────────────────────┘
```

---

# 🖼️ Architecture Diagram

<img width="1024" height="572" alt="architecture" src="https://github.com/user-attachments/assets/1d0d33fc-ada6-47b4-97b0-04fcc25d1bb5" />



## 🔄 End-to-End RAG Workflow
```text
The complete application flow is divided into two major stages.
```
---

## 1️⃣ Offline / Indexing Pipeline

```text
Policy PDF
    ↓
PDF Loader
    ↓
Text Extraction
    ↓
Document Chunking
    ↓
Metadata Creation
    ↓
OpenAI Embeddings
    ↓
ChromaDB
```

The documents are processed and converted into vector representations.

---

## 2️⃣ Online / Query Pipeline

```text
User Query
    ↓
Query Classification
    ↓
Query Rewriting
    ↓
Semantic Retrieval
    ↓
LLM Reranking
    ↓
Relevance Guard
    ↓
Context Builder
    ↓
Prompt Builder
    ↓
OpenAI LLM
    ↓
Grounding Guard
    ↓
Groundedness Guard
    ↓
Answer Retry if Required
    ↓
Final Answer
    ↓
Source Information
```

---

# 📂 Project Structure

```text
SBI_Insurance_RAG/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   └── .gitkeep
│
├── ingestion/
│   ├── __init__.py
│   └── pdf_loader.py
│
├── rag/
│   ├── __init__.py
│   └── indexing_pipeline.py
│
├── vectorstore/
│   ├── __init__.py
│   └── chroma_store.py
│
├── retrieval/
│   ├── __init__.py
│   └── semantic_retriever.py
│
├── reranking/
│   ├── __init__.py
│   └── llm_reranker.py
│
├── context/
│   ├── __init__.py
│   └── context_builder.py
│
├── prompts/
│   ├── __init__.py
│   └── prompt_builder.py
│
├── llm/
│   ├── __init__.py
│   └── openai_llm.py
│
├── guards/
│   ├── __init__.py
│   ├── relevance_guard.py
│   ├── grounding_guard.py
│   ├── groundedness_guard.py
│   └── answer_retry_guard.py
│
├── conversation/
│   ├── __init__.py
│   └── conversation_manager.py
│
├── services/
│   ├── __init__.py
│   └── rag_service.py
│
├── scripts/
│   ├── check_environment.py
│   ├── test_ingestion.py
│   ├── test_indexing.py
│   ├── test_retrieval.py
│   ├── test_query_rewriter.py
│   ├── test_reranker.py
│   ├── test_context_builder.py
│   ├── test_llm.py
│   ├── test_rag_service.py
│   └── test_groundedness_guard.py
│
├── screenshots/
│   ├── architecture.png
│   ├── 01_home_screen.png
│   ├── 02_policy_selection.png
│   ├── 03_user_query.png
│   ├── 04_generated_answer.png
│   ├── 05_sources.png
│   └── 06_conversation.png
│
└── chroma_db/
    └── .gitkeep
```

---

# 📄 Document Ingestion Pipeline

The ingestion pipeline loads the insurance policy PDFs and converts them into documents that can be processed by the RAG system.

```text
PDF Documents
      ↓
PDF Loader
      ↓
Page-wise Text Extraction
      ↓
Document Objects
      ↓
Metadata
```

Metadata such as:

* Policy type
* Document name
* Page number
* Source

is preserved throughout the pipeline.

This metadata is later used for source attribution.

---

# 🗂️ Indexing Pipeline

After document ingestion, the extracted content is converted into vector representations.

```text
Document Chunks
      ↓
OpenAI Embedding Model
      ↓
1536-dimensional Embeddings
      ↓
ChromaDB
```

The embedding model used in the project is:

```text
text-embedding-3-small
```

The generated embeddings are stored in ChromaDB for semantic retrieval.

---

# 🔎 Query Processing Pipeline

When the user submits a question, the query passes through multiple stages.

## Step 1 — Query Classification

The system first determines the type and relevance of the incoming query.

---

## Step 2 — Query Rewriting

The original query can be rewritten into a retrieval-friendly query.

Example:

```text
Original Query:
What medical expenses are covered?

        ↓

Rewritten Query:
What medical expenses and medical treatment costs
are covered under the travel insurance policy?
```

The goal is to improve semantic retrieval.

---

# 🔍 Semantic Retrieval

The rewritten query is converted into an embedding and compared with the indexed document vectors.

```text
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Top-K Relevant Chunks
```

Policy filtering can also be applied so that retrieval focuses on the selected insurance policy.

---

# 🎯 LLM-Based Reranking

Initial vector retrieval may return several potentially relevant chunks.

The application therefore performs a second-stage reranking process.

```text
Retrieved Documents
        ↓
      Reranker
        ↓
Relevance Evaluation
        ↓
Top Relevant Documents
```

This improves the quality of the context provided to the LLM.

---

# 🧩 Context Builder

The selected documents are converted into structured context.

```text
Retrieved Documents
        ↓
Metadata Extraction
        ↓
Context Construction
        ↓
Prompt Context
```

The context includes relevant information such as:

```text
Policy Type
Document Name
Page Number
Source
Relevant Text
```

---

# 🛡️ RAG Guardrails

One of the key aspects of this project is the use of multiple validation mechanisms.

---

## 1️⃣ Relevance Guard

The relevance guard checks whether the retrieved information is sufficiently relevant to the user's question.

```text
User Query
     ↓
Retrieved Context
     ↓
Relevance Validation
     ↓
Relevant?
   /     \
 Yes      No
  ↓        ↓
Continue   Stop / Handle
```

This helps prevent irrelevant context from being passed into the generation stage.

---

## 2️⃣ Grounding Guard

The grounding guard validates whether the generated answer is supported by the retrieved policy context.

The objective is to prevent the model from introducing unsupported information.

```text
Retrieved Context
       +
Generated Answer
       ↓
Grounding Validation
       ↓
Supported?
```

---

## 3️⃣ Groundedness Guard

The groundedness guard performs another validation of the generated answer against the supplied policy context.

It checks that the answer does not introduce unsupported:

* Benefits
* Coverage
* Monetary values
* Conditions
* Policy information
* Claims

that are not present in the retrieved context.

---

## 4️⃣ Answer Retry Guard

If answer validation fails, the application can retry generation using stricter instructions.

```text
                Generated Answer
                       ↓
                  Validation
                       ↓
                 ┌─────┴─────┐
                 │           │
                PASS        FAIL
                 │           │
                 ▼           ▼
             Final Answer   Retry
                               │
                               ▼
                              LLM
                               │
                               ▼
                           Validation
```

This provides an additional layer of protection against unsupported responses.

---

# 💬 Conversation Management

The application supports conversational interactions using a conversation manager.

Example:

```text
User:
What medical expenses are covered?

Assistant:
[Answer]

User:
What about hospitalization?

Assistant:
[Context-aware answer]
```

Conversation history is maintained so that follow-up questions can be handled more effectively.

---

# 🖥️ Streamlit Application

The project includes a Streamlit-based user interface.

The UI allows users to interact with the RAG system without directly interacting with the underlying Python components.

The application provides:

* Policy interaction
* Natural-language question input
* Generated answers
* Source information
* Follow-up questions
* Conversation history

---

# 📸 Application Screenshots

## 1️⃣ Application Home Screen

<img width="1920" height="1080" alt="main_ui" src="https://github.com/user-attachments/assets/0de8f681-81be-43ba-aeee-50c24353115c" />


## 2️⃣ Policy Selection & 3️⃣ User Query

<img width="1920" height="1080" alt="UI_Conversation" src="https://github.com/user-attachments/assets/ec07f6ce-ef58-4b39-a1ff-97291fd8b659" />


## 4️⃣ Generated Answer

Example query:

<img width="1920" height="1080" alt="UI_Pipeline" src="https://github.com/user-attachments/assets/2c050fdc-59e0-4034-9f08-c7198e8a2b97" />

```text
What medical expenses are covered under the insurance policy?
```

## 5️⃣ Source Information & 6️⃣ Conversation / Follow-Up Question

<img width="1920" height="1080" alt="UI_Pipeline" src="https://github.com/user-attachments/assets/49f155c1-9480-47b0-adf9-0c7f72423ee0" />

---

# 🧪 Testing

The project contains individual test scripts for validating each major component.

## Environment Test

```powershell
python -m scripts.check_environment
```

---

## Ingestion Test

```powershell
python -m scripts.test_ingestion
```

---

## Indexing Test

```powershell
python -m scripts.test_indexing
```

---

## Retrieval Test

```powershell
python -m scripts.test_retrieval
```

---

## Query Rewriter Test

```powershell
python -m scripts.test_query_rewriter
```

---

## Reranker Test

```powershell
python -m scripts.test_reranker
```

---

## Context Builder Test

```powershell
python -m scripts.test_context_builder
```

---

## LLM Test

```powershell
python -m scripts.test_llm
```

---

## Groundedness Guard Test

```powershell
python -m scripts.test_groundedness_guard
```

---

## End-to-End RAG Test

```powershell
python -m scripts.test_rag_service
```

---

# 📊 Project Results

The completed indexing pipeline produced the following results during development:

| Metric                    |                   Result |
| ------------------------- | -----------------------: |
| Total PDF pages processed |                      199 |
| Chunks created            |                      938 |
| Vectors stored            |                      938 |
| Embedding model           | `text-embedding-3-small` |
| Embedding dimensions      |                     1536 |
| Vector database           |                 ChromaDB |
| LLM                       |            `gpt-4o-mini` |

The end-to-end RAG pipeline was successfully tested through:

```text
Query
 ↓
Classification
 ↓
Query Rewriting
 ↓
Retrieval
 ↓
Reranking
 ↓
Relevance Validation
 ↓
Context Construction
 ↓
Prompt Construction
 ↓
LLM Generation
 ↓
Grounding Validation
 ↓
Groundedness Validation
 ↓
Final Answer
```

---

# 🛠️ Technology Stack

| Technology             | Purpose                                |
| ---------------------- | -------------------------------------- |
| Python                 | Core programming language              |
| LangChain              | RAG orchestration                      |
| OpenAI                 | Embeddings and LLM                     |
| GPT-4o-mini            | Answer generation                      |
| text-embedding-3-small | Text embeddings                        |
| ChromaDB               | Vector database                        |
| Streamlit              | User interface                         |
| PyPDF                  | PDF document processing                |
| Git/GitHub             | Version control and repository hosting |

---

# ⚙️ Installation

## 1. Clone the Repository

```powershell
git clone https://github.com/knagaraj-2106/SBI-Insurance-Policy-RAG-Assistant.git

cd SBI_Insurance_RAG
```

---

## 2. Create Virtual Environment

```powershell
python -m venv sbi_insurance_policy
```

Activate the environment:

```powershell
.\sbi_insurance_policy\Scripts\activate
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

The repository provides a `.env.example` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Copy `.env.example` to `.env` and add your own OpenAI API key.

Example:

```env
OPENAI_API_KEY=your_actual_api_key
```

### ⚠️ Security

Never commit your actual `.env` file or API key to GitHub.

The `.env` file should be excluded using `.gitignore`.

---

# ▶️ Running the Application

## Step 1 — Activate Environment

```powershell
.\sbi_insurance_policy\Scripts\activate
```

---

## Step 2 — Build / Verify the Index

Run the indexing pipeline:

```powershell
python -m scripts.test_indexing
```

---

## Step 3 — Start Streamlit

```powershell
streamlit run app.py
```

The Streamlit application will then open in your browser.

---

# 🔄 Complete Application Flow

```text
                   ┌───────────────────┐
                   │ Insurance PDF Docs │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ Document Ingestion│
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │     Chunking      │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ OpenAI Embeddings │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │     ChromaDB      │
                   └─────────┬─────────┘
                             │
                             │
                       User Query
                             │
                             ▼
                   ┌───────────────────┐
                   │ Query Classifier  │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ Query Rewriter    │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ Semantic Retrieval│
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │  LLM Reranking    │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │  Relevance Guard  │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ Context Builder   │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ Prompt Builder    │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │    OpenAI LLM     │
                   └─────────┬─────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ Grounding + Groundedness│
                │        Validation        │
                └────────────┬─────────────┘
                             │
                     ┌───────┴───────┐
                     │               │
                    PASS            FAIL
                     │               │
                     ▼               ▼
                 Answer           Retry
                     │               │
                     └───────┬───────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │ Answer + Sources  │
                   └───────────────────┘
```

---

# 🚀 Production Considerations

Although this project is implemented as a portfolio/learning application, the architecture incorporates several production-oriented concepts.

Potential production deployment could introduce:

### Scalability

* Multiple application instances
* Load balancing
* Distributed vector databases
* Horizontal scaling

### Performance

* Embedding caching
* Retrieval caching
* LLM response caching
* Asynchronous processing

### Reliability

* Retry policies
* Timeout handling
* Error handling
* Monitoring
* Logging

### Security

* Secret management
* API authentication
* Authorization
* Input validation
* PII protection

### Observability

* Request tracing
* Retrieval metrics
* Latency monitoring
* LLM token usage
* Error monitoring
* RAG evaluation metrics

---

# 🔮 Future Enhancements

Possible future improvements include:

* Hybrid search using semantic + keyword retrieval
* Cross-encoder reranking
* Advanced chunking strategies
* Parent-child retrieval
* Multi-query retrieval
* Query decomposition
* Reciprocal Rank Fusion
* RAG evaluation using automated evaluation frameworks
* Retrieval precision/recall evaluation
* Answer faithfulness evaluation
* LLM observability
* FastAPI backend
* Docker deployment
* Cloud deployment
* Authentication and authorization
* Production vector database
* Asynchronous document ingestion
* Background processing using queues
* CI/CD pipeline

---

# 📚 Learning Outcomes

This project provided hands-on experience with:

* Building an end-to-end RAG pipeline
* PDF document ingestion
* Document chunking
* Embedding generation
* Vector databases
* Semantic retrieval
* Query rewriting
* LLM-based reranking
* Context construction
* Prompt engineering
* Grounded answer generation
* RAG guardrails
* Answer validation
* Retry mechanisms
* Conversational memory
* Source attribution
* Streamlit application development
* Modular Python architecture
* Component-level testing
* End-to-end RAG testing

---

# 📈 Key Architecture Highlights

The main difference between this implementation and a basic RAG chatbot is the addition of validation and quality-control layers.

### Basic RAG

```text
Query
 ↓
Retrieval
 ↓
LLM
 ↓
Answer
```

### This Project

```text
Query
 ↓
Classification
 ↓
Query Rewriting
 ↓
Retrieval
 ↓
Reranking
 ↓
Relevance Validation
 ↓
Context Construction
 ↓
LLM
 ↓
Grounding Validation
 ↓
Groundedness Validation
 ↓
Retry if Required
 ↓
Answer + Sources
```

This provides a more robust architecture for policy-oriented question answering.

---

# ⚠️ Disclaimer

This project is intended for **educational, demonstration, and portfolio purposes**.

It should not be treated as a replacement for official insurance policy documentation, professional advice, or official claim/coverage decisions.

Users should always refer to the applicable official policy documents and terms and conditions for authoritative information.

---

# 👨‍💻 Author

**Nagaraj Kamale**

Generative AI Engineer | Machine Learning Engineer

### Areas of Interest

* Generative AI
* Retrieval-Augmented Generation
* LLM Applications
* LangChain
* Prompt Engineering
* AI Agents
* Machine Learning
* Document Intelligence
* Production AI Systems

---

# ⭐ If you find this project useful

Feel free to explore the repository, review the architecture, and experiment with the RAG pipeline.

**Built with Python + LangChain + OpenAI + ChromaDB + Streamlit 🚀**
