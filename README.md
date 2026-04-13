# 🧠 Local MAS (Multi-Agent System) Orchestrator

> **Status:** Active prototyping & internal tooling.
> **Philosophy:** *We are past the chatbot days. This is actual local AI orchestration.*

## 🚀 Overview
An asynchronous Multi-Agent System (MAS) designed to run perfectly on deeply constrained hardware (8GB RAM). 

Most AI implementations fail because they rely on massive cloud infrastructure and single-prompt magic. This project solves that by breaking down complex developer workflows into modular, dedicated local agents that run sequentially or asynchronously, managed by a central orchestrator.

### 📊 System Architecture Flow

```mermaid
graph TD
    subgraph "📡 Data Ingestion"
        A["📧 IMAP (Gmail Inbox)"] --> C
        B["🌐 RSS Feeds / Webhooks"] --> C
    end

    subgraph "⚡ Core Orchestration (Asyncio)"
        C{"⚙️ MAS Central Orchestrator"}
        C -->|Raw Data| D["🧠 Context Manager (8GB Limit Optimizer)"]
        D -->|Optimized Payload| E["🤖 Local LLM Engine (Claude/Ollama)"]
        E -->|Structured JSON Output| C
    end

    subgraph "🚀 Output Execution"
        C -->|Decision: Approved| F["📲 Telegram Alert Agent"]
        C -->|Decision: Rejected| G["🗑️ Local Logs / SQLite"]
    end

    %% Premium Dark Styling
    style C fill:#0f3460,stroke:#e94560,stroke-width:2px,color:#ffffff
    style D fill:#1a1a2e,stroke:#4a4e69,stroke-width:1px,color:#e4e4e4
    style E fill:#1a1a2e,stroke:#4a4e69,stroke-width:1px,color:#e4e4e4
    style A fill:#16213e,stroke:#e94560,color:#fff
    style B fill:#16213e,stroke:#e94560,color:#fff
    style F fill:#00d2ff,stroke:#fff,color:#000
    style G fill:#111,stroke:#333,color:#888
```

🏗 System Architecture
The Brain (Orchestrator): mas_orchestrator.py - Manages agent lifecycle, task assignment, and context window limits.
Hardware First: Built to bypass the memory limits of commercial systems. Context is passed strictly via minimal JSON payloads, never dumping entire codebases.
Asynchronous I/O: Built natively on asyncio to handle multiple LLM API streams without freezing the main event loop.
💼 Why this matters (Business Outcome)
This is the "glue" between unstructured AI models and actual business systems. By orchestrating distinct agents (e.g., a Researcher, a Coder, a Reviewer) locally, we eliminate hallucination loops and ship production-ready automations faster than traditional software engineering cycles.

🛠 Tech Stack
Python 3.11+
Asyncio / Aiohttp
Local AI Context Management
