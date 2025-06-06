# Custom instructions for Copilot

## Project Context
This project is an **MCP (Model Context Protocol) Server designed as a Physics Problem Solver**. It will:
1. Ingest physics knowledge from books (PDFs/text).
2. Build a structured, non-redundant knowledge base of equations, concepts, and definitions.
3. Utilize the Google Gemini LLM for parsing, making connections, and understanding queries.
4. Employ symbolic math tools (like SymPy) to solve problems based on retrieved knowledge.
5. Provide reasoned solutions and explanations via MCP.

## Preferred Technologies/Frameworks
- **Primary Language:** Python 3.10 or higher
- **MCP Framework:** MCP Python SDK (using FastMCP)
- **LLM:** Google Gemini (via `google-generativeai` Python library)
- **Symbolic Math:** SymPy
- **PDF Processing (Optional):** PyPDF2 or similar
- **Database (Consideration for later):** SQLite (initially in-memory Python dictionaries)

## General Coding Style
- All Python functions should have clear type hints and descriptive docstrings.
- Code should be modular, separating concerns for knowledge base management, LLM interaction, symbolic solving, and MCP primitives.
- Equations should be stored and handled in LaTeX format where possible.
- Concepts and definitions should be clearly linked to their sources and related items.
- Strive for non-redundancy in the knowledge base.

## Testing Preferences
- Unit tests (e.g., using `unittest` or `pytest`) should be written for:
    - Knowledge ingestion and structuring logic.
    - Individual components of the problem-solving pipeline (e.g., equation retrieval, symbolic solver integration).
    - MCP tool and resource handlers.
- Integration tests will be important to verify the end-to-end problem-solving flow.

## Specific Instructions for Copilot
- When generating MCP tool handlers, ensure they correctly use `FastMCP` decorators and handle input/output schemas (e.g., using Pydantic models).
- When suggesting code for LLM interaction with Gemini, use the `google-generativeai` library and best practices for prompt engineering (e.g., clear instructions, few-shot examples if applicable).
- For symbolic math, generate code compatible with the SymPy library.
- When interacting with the knowledge base (even if initially in-memory), suggest code that is organized and allows for future transition to a database.
- Prioritize clarity and precision in physics-related explanations.