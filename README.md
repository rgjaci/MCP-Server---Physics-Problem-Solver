# 🧠 MCP Server – Model Context Protocol for Physics

> A physics-aware reasoning server that learns from books, stores equations as memory, and solves problems by predicting and applying the right formulas—like an LLM, but with logic.

---

## 📌 Overview

The **MCP (Model Context Protocol)** server is an intelligent system designed to solve physics problems by building a structured memory of physics equations, definitions, and concepts. It learns from books—PDFs or text—using the Google Gemini language model to extract, deduplicate, and interconnect key knowledge.

When given a physics question, the MCP server:
1.  Predicts relevant formulas and concepts from its memory.
2.  Retrieves and applies them with a symbolic equation-solving engine (SymPy).
3.  Produces a precise, explainable solution via MCP.

Unlike traditional LLMs that generate text token by token, this MCP server combines:
-   LLM-powered semantic understanding (Google Gemini),
-   A structured knowledge base (initially in-memory, potentially vector-based retrieval later), and
-   Logic-driven symbolic math (SymPy).

---

## 🚀 Features

-   📚 **Learn from Books**: Ingest textbooks (PDF/text) and extract meaningful formulas, definitions, and concepts.
-   🧠 **Knowledge Base**: Structured, queryable store of learned physics knowledge.
-   🔍 **LLM-Powered Analysis**: Use Google Gemini to interpret input problems and extract new knowledge.
-   🧮 **Symbolic Equation Solving**: Use `SymPy` for precise problem solving (unit handling with `Pint` can be integrated).
-   🤖 **MCP Interface**: Expose capabilities as MCP tools and resources.
-   🗣️ **Explainable Solutions**: Provide step-by-step reasoning for solved problems.
-   *(Future)* **Embedding-Based Retrieval**: Match new problems to known equations with vector search.
-   *(Future)* **Human-in-the-loop Editing**: UI to review and edit extracted knowledge.

---

## 🏗️ Project Structure

MCP Server - Physics Problem Solver/
│
├── .venv/                  # Virtual environment (typically in .gitignore)
├── server.py               # Main MCP server logic, tool & resource definitions
├── sample_physics.txt      # Example input file for ingestion
├── README.md               # This file
├── .env                    # Environment variables (e.g., GEMINI_API_KEY)
├── requirements.txt        # Python dependencies (to be created/updated)
└── .instructions.md        # Copilot project instructions (if used)

---

## ⚙️ Getting Started

### 1. Setup Environment

If you haven't already, create and activate a Python virtual environment:

```bash
# Navigate to your project directory
# cd "MCP Server - Physics Problem Solver"

python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
# source .venv/bin/activate
```

### 2. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
python-dotenv
google-generativeai
fastmcp
sympy
# Add other dependencies like PyPDF2 if you implement PDF parsing
```

Then install them:

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root and add your Google Gemini API key:

```env
GEMINI_API_KEY="your-gemini-api-key-here"
```

### 4. Run the MCP Server

Execute the `server.py` script:

```bash
python server.py
```

The server will start and listen for MCP messages via standard input/output (stdio). You can interact with it using an MCP client like the MCP Inspector.

---

## 🧪 Example Interaction

**Tool: `ingest_document`**
*   **Params**:
    ```json
    {
      "document_content": "Newton's second law states that Force = mass * acceleration. F=m*a.",
      "source_identifier": "Newton's Second Law Note"
    }
    ```
*   **Server will**:
    *   Use Gemini to parse the text.
    *   Extract the equation `F=m*a` and the concept of "Newton's second law".
    *   Store them in its in-memory knowledge base.

**Tool: `solve_physics_problem`**
*   **Params**:
    ```json
    {
      "question": "An object of mass 2 kg accelerates at 3 m/s². What is the force?"
    }
    ```
*   **Server will**:
    *   Use Gemini to analyze the question and identify relevant concepts/equations (e.g., `F=m*a`).
    *   (Conceptually) Use SymPy to substitute values and solve.
    *   Return a structured response including the identified concepts, equations, steps, and the final answer (e.g., `F = 6 N`).

---

## 🛠️ Roadmap

-   [x] Basic question parsing and LLM-driven analysis.
-   [x] Initial document ingestion (text-based).
-   [ ] Robust PDF textbook ingestion.
-   [ ] Enhanced symbolic solving with SymPy for a wider range of problems.
-   [ ] Embedding-based retrieval for more scalable knowledge lookup.
-   [ ] Development of a concept dependency graph.
-   [ ] Unit and integration tests for all components.
-   [ ] *(Optional)* Human-in-the-loop UI for knowledge base editing.
-   [ ] *(Future)* Multilingual physics understanding.

---

## 📄 License

MIT License

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## ✨ Credits

-   Created by Reison Gjaci
-   Powered by Google Gemini, SymPy, FastMCP, and Python
