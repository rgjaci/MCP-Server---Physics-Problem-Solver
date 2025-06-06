from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import os # For API key, though we'll load .env below
import json # For parsing Gemini's output
from google import genai
from dotenv import load_dotenv # To load .env file
import sympy
import time # Added for timing logs

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None # Initialize to None
gemini_model = None # Initialize to None
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set in .env file. Ingestion and solving tools will be limited.")
else:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        gemini_model = client.models.get(model="gemini-2.5-flash-preview-05-20") # Using the latest flash model
        print("Gemini API configured successfully with model:", gemini_model.name)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        GEMINI_API_KEY = None # Ensure it's None if config fails
        client = None # Reset client if config fails
        gemini_model = None # Reset model if config fails

# Create the FastMCP server instance
mcp = FastMCP(
    title="Physics Knowledge Server",
    description="A server for ingesting and querying physics knowledge.",
    version="0.1.0",
)

# --- Knowledge Base (Initial Placeholder) ---
knowledge_base = {
    "equations": {},
    "concepts": {},
    "documents": {}
}
print("Knowledge base initialized (in-memory).")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Placeholder for PDF text extraction logic.
    Currently handles .txt files and returns placeholder for .pdf.
    """
    print(f"Attempting to extract text from: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Error: Document path does not exist: {pdf_path}")
        return ""

    if pdf_path.lower().endswith(".txt"):
        try:
            with open(pdf_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file {pdf_path}: {e}")
            return ""
    elif pdf_path.lower().endswith(".pdf"):
        # In a real implementation, use PyPDF2 or another library here
        # For now, a placeholder:
        print("PDF processing placeholder: Returning sample content.")
        # To make this testable, let's return some mock physics content
        return "This is sample PDF content. Newton's first law states that an object at rest stays at rest and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force. E=mc^2 relates energy (E) to mass (m) and the speed of light (c)."
    else:
        print(f"Unsupported file type: {pdf_path}")
        return ""

# --- Pydantic Models for ingest_document tool ---
class IngestDocumentInput(BaseModel):
    document_path: str | None = Field(None, description="Path to the PDF or text file to ingest.")
    document_content: str | None = Field(None, description="Direct text content of the document.")
    source_identifier: str = Field(description="A unique identifier for the document, e.g., book title and edition.")

class IngestDocumentOutput(BaseModel):
    status: str
    summary: str
    new_equations_added: int
    new_concepts_added: int


# --- Pydantic Models for solve_physics_problem tool ---
class SolvePhysicsProblemInput(BaseModel):
    question: str = Field(description="The physics question to be solved.")

class SolvePhysicsProblemOutput(BaseModel):
    original_question: str
    identified_concepts: list[str] = Field(default_factory=list)
    relevant_equations_latex: list[str] = Field(default_factory=list) # Store LaTeX for clarity
    symbolic_solution_steps: str | None = None # To show steps or the result
    final_answer: str | None = None # A concise final answer if calculable
    explanation: str
    status: str # e.g., "solved", "needs_more_info", "error", "explanation_provided"


# --- MCP Tool for Ingesting Documents ---
@mcp.tool()
async def ingest_document(params: IngestDocumentInput) -> IngestDocumentOutput:
    """
    Ingests a physics document (PDF or text), extracts knowledge,
    and updates the server's knowledge base.
    """
    if not client or not gemini_model: # Check if gemini_model was configured
        return IngestDocumentOutput(status="error", summary="Gemini API not configured or failed to initialize.", new_equations_added=0, new_concepts_added=0)

    print(f"Ingesting document: {params.source_identifier}")
    content_to_process = ""
    if params.document_path:
        content_to_process = extract_text_from_pdf(params.document_path)
    elif params.document_content:
        content_to_process = params.document_content
    else:
        return IngestDocumentOutput(status="error", summary="No document path or content provided.", new_equations_added=0, new_concepts_added=0)

    if not content_to_process:
        return IngestDocumentOutput(status="error", summary="Failed to extract content or content is empty.", new_equations_added=0, new_concepts_added=0)

    # --- Gemini for Extraction & Structuring ---
    # Ensure content_to_process is not overly long for a single API call.
    # Real implementation needs chunking for large documents.
    max_prompt_length = 15000 # Adjust based on Gemini model limits & typical content size
    truncated_content = content_to_process[:max_prompt_length]
    if len(content_to_process) > max_prompt_length:
        print(f"Warning: Content for '{params.source_identifier}' was truncated to {max_prompt_length} characters for Gemini prompt.")


    prompt = f"""
    You are a physics knowledge extraction system.
    Given the following text from '{params.source_identifier}', identify and extract:
    1. Key equations (provide in LaTeX format, list variables involved with brief descriptions, and a general description of the equation).
    2. Core concepts (provide a name and a concise definition for each).
    3. Important definitions of specific terms (provide the term and its explanation).

    Structure your output as a valid JSON object with top-level keys "equations", "concepts", "definitions".
    Each item in "equations" should be an object with "latex", "variables" (an array of objects, each with "symbol" and "description"), and "description".
    Each item in "concepts" should be an object with "name" and "definition".
    Each item in "definitions" should be an object with "term" and "explanation".

    Text:
    ---
    {truncated_content}
    ---
    JSON Output:
    """

    try:
        print(f"Sending prompt to Gemini for '{params.source_identifier}'...")
        response = await client.aio.models.generate_content(
            model=gemini_model.name,
            contents=prompt,
            # Optional: To enforce JSON output structure if Gemini supports it strongly,
            # you might use generation_config with response_mime_type and response_schema.
            # For now, relying on prompt for JSON structure.
            # generation_config=genai.types.GenerationConfig(
            # response_mime_type="application/json"

        )
        
        # It's good practice to log the raw response for debugging
        # print(f"Raw Gemini response: {response.text}")

        # Clean potential markdown and extract JSON
        json_response_text = response.text.strip()
        if json_response_text.startswith("```json"):
            json_response_text = json_response_text[len("```json"):]
        if json_response_text.endswith("```"):
            json_response_text = json_response_text[:-len("```")]
        json_response_text = json_response_text.strip()
        
        extracted_data = json.loads(json_response_text)
        
        # --- Update Knowledge Base (Simplified In-Memory) ---
        doc_id = params.source_identifier # Use a more robust ID in a real system
        knowledge_base["documents"][doc_id] = {"title": params.source_identifier, "status": "processed"}
        
        eq_added_count = 0
        for eq_data in extracted_data.get("equations", []):
            if isinstance(eq_data, dict) and "latex" in eq_data:
                eq_id = f"eq_{hash(eq_data['latex'])}" # Simple ID based on LaTeX hash
                knowledge_base["equations"][eq_id] = eq_data
                eq_added_count += 1
            else:
                print(f"Warning: Skipping malformed equation data: {eq_data}")

        concept_added_count = 0
        for concept_data in extracted_data.get("concepts", []):
            if isinstance(concept_data, dict) and "name" in concept_data:
                concept_id = f"concept_{hash(concept_data['name'])}"
                knowledge_base["concepts"][concept_id] = concept_data
                concept_added_count += 1
            else:
                print(f"Warning: Skipping malformed concept data: {concept_data}")
        
        for def_data in extracted_data.get("definitions", []):
            if isinstance(def_data, dict) and "term" in def_data:
                # Could merge definitions into concepts or keep separate
                def_id = f"definition_{hash(def_data['term'])}"
                knowledge_base["concepts"][def_id] = {"name": def_data["term"], "definition": def_data["explanation"], "type": "definition"}
                concept_added_count += 1
            else:
                print(f"Warning: Skipping malformed definition data: {def_data}")

        summary = f"Successfully processed '{params.source_identifier}'. Added {eq_added_count} equations, {concept_added_count} concepts/definitions."
        print(summary)
        # For debugging, print the current state of the knowledge base
        # print(f"Current Equations: {knowledge_base['equations']}")
        # print(f"Current Concepts: {knowledge_base['concepts']}")

        return IngestDocumentOutput(
            status="success",
            summary=summary,
            new_equations_added=eq_added_count,
            new_concepts_added=concept_added_count
        )

    except json.JSONDecodeError as json_e:
        response_text_snippet = response.text[:500] if "response" in locals() and hasattr(response, 'text') else "N/A"
        error_summary = f"Error decoding JSON from Gemini: {json_e}. Response text: '{response_text_snippet}...'"
        print(error_summary)
        return IngestDocumentOutput(status="error", summary=error_summary, new_equations_added=0, new_concepts_added=0)
  
    except Exception as e:
        error_summary = f"Error during ingestion with Gemini: {e}"
        print(error_summary)
        return IngestDocumentOutput(status="error", summary=error_summary, new_equations_added=0, new_concepts_added=0)


# --- MCP Resource Implementations ---
@mcp.resource("physicsdb://documents/list")
async def list_documents_resource() -> list[dict]:
    """Lists all documents currently processed into the knowledge base."""
    print("Request received for: physicsdb://documents/list")
    if not knowledge_base["documents"]:
        return [{"id": "none", "title": "No documents processed yet."}]
    
    return [{"id": doc_id, "title": data.get("title", "Unknown Title")} 
            for doc_id, data in knowledge_base["documents"].items()]

@mcp.resource("physicsdb://equations/list")
async def list_equations_resource() -> list[dict]:
    """Lists all equations currently in the knowledge base."""
    print("Request received for: physicsdb://equations/list")
    if not knowledge_base["equations"]:
        return [{"id": "none", "description": "No equations available."}]

    # Return a summary: id and description
    return [{"id": eq_id, "description": data.get("description", "No description"), "latex": data.get("latex", "")}
            for eq_id, data in knowledge_base["equations"].items()]

@mcp.resource("physicsdb://equations/{equation_id}")
async def get_equation_resource(equation_id: str) -> dict:
    """Provides the details of a specific equation by its ID."""
    print(f"Request received for: physicsdb://equations/{equation_id}")
    equation_data = knowledge_base["equations"].get(equation_id)
    if not equation_data:
        # In a real application, you might raise a specific MCP error
        # or return a standardized error structure.
        # For simplicity, we return a dictionary indicating not found.
        return {"error": "not_found", "message": f"Equation with ID '{equation_id}' not found."}
    return equation_data

@mcp.resource("physicsdb://concepts/list")
async def list_concepts_resource() -> list[dict]:
    """Lists all concepts (and definitions) currently in the knowledge base."""
    print("Request received for: physicsdb://concepts/list")
    if not knowledge_base["concepts"]:
        return [{"id": "none", "name": "No concepts available."}]
        
    # Return a summary: id and name
    return [{"id": concept_id, "name": data.get("name", "Unknown Concept"), "type": data.get("type", "concept")}
            for concept_id, data in knowledge_base["concepts"].items()]

@mcp.resource("physicsdb://concepts/{concept_id}")
async def get_concept_resource(concept_id: str) -> dict:
    """Provides the details of a specific concept by its ID."""
    print(f"Request received for: physicsdb://concepts/{concept_id}")
    concept_data = knowledge_base["concepts"].get(concept_id)
    if not concept_data:
        return {"error": "not_found", "message": f"Concept with ID '{concept_id}' not found."}
    return concept_data

# --- MCP Tool for Solving Physics Problems ---
@mcp.tool()
async def solve_physics_problem(params: SolvePhysicsProblemInput) -> SolvePhysicsProblemOutput:
    """
    Solves a given physics problem using the internal knowledge base,
    symbolic math (conceptually), and LLM reasoning.
    """
    question = params.question
    print(f"Received physics problem: {question}")

    if not client or not gemini_model:
        return SolvePhysicsProblemOutput(
            original_question=question,
            explanation="Gemini API not configured or failed to initialize.",
            status="error"
        )

    # --- Stage 1: Gemini for Query Analysis & Retrieval Plan ---
    # Create a summary of available knowledge for the prompt
    # This helps Gemini know what it *can* look for.
    # For a large KB, this summary would need to be more sophisticated (e.g., embeddings-based RAG)
    
    concept_summaries = [f"- {data.get('name', cid)}: {data.get('definition', '')[:100]}..." # Truncate definition
                         for cid, data in list(knowledge_base["concepts"].items())[:15]] # Limit for prompt
    equation_summaries = [f"- {data.get('description', eid)} ({data.get('latex', '')})"
                          for eid, data in list(knowledge_base["equations"].items())[:15]] # Limit for prompt

    available_knowledge_prompt_text = "Available Concepts Summary:\n" + "\n".join(concept_summaries) + \
                                     "\n\nAvailable Equations Summary:\n" + "\n".join(equation_summaries)

    prompt_retrieve = f"""
    Analyze the following physics question: "{question}"

    Based on the question and the provided summary of available knowledge, identify:
    1. Key physics concepts involved (list their names).
    2. Specific equations (by their LaTeX or description) that are likely needed for the solution.
    3. Any numerical values provided in the question along with their associated variables/units if identifiable.
    4. What is the primary unknown variable or quantity to be solved for?

    Available Knowledge Summary:
    ---
    {available_knowledge_prompt_text}
    ---

    Return your analysis as a VALID JSON object with keys:
    "identified_concept_names": ["list of names"],
    "relevant_equation_signatures": ["list of LaTeX or descriptions of equations"],
    "given_values": [{{"variable_guess": "symbol", "value": "number", "unit": "unit_symbol"}}],
    "target_unknown": "variable_or_quantity_description"
    
    JSON Output:
    """

    retrieved_concept_names = []
    retrieved_equations_latex = []
    given_values_parsed = []
    target_unknown_parsed = "Not identified"
    retrieval_duration = 0

    try:
        print("Asking Gemini for retrieval plan...")
        retrieval_start_time = time.time()
        response_retrieve = await client.aio.models.generate_content(model=gemini_model.name, contents=prompt_retrieve)
        retrieval_end_time = time.time()
        retrieval_duration = retrieval_end_time - retrieval_start_time
        print(f"DEBUG: Gemini retrieval plan call took {retrieval_duration:.2f} seconds.")

        retrieval_plan_json_text = response_retrieve.text.strip()
        if retrieval_plan_json_text.startswith("```json"):
            retrieval_plan_json_text = retrieval_plan_json_text[len("```json"):]
        if retrieval_plan_json_text.endswith("```"):
            retrieval_plan_json_text = retrieval_plan_json_text[:-len("```")]
        retrieval_plan_json_text = retrieval_plan_json_text.strip()
        
        retrieval_plan = json.loads(retrieval_plan_json_text)
        
        retrieved_concept_names = retrieval_plan.get("identified_concept_names", [])
        relevant_equation_signatures = retrieval_plan.get("relevant_equation_signatures", [])
        given_values_parsed = retrieval_plan.get("given_values", [])
        target_unknown_parsed = retrieval_plan.get("target_unknown", "Not explicitly identified")

        # Fetch full equation details from knowledge_base based on signatures
        for sig in relevant_equation_signatures:
            for eq_data in knowledge_base["equations"].values():
                if sig in eq_data.get("latex", "") or sig in eq_data.get("description", ""):
                    if eq_data.get("latex") not in retrieved_equations_latex:
                         retrieved_equations_latex.append(eq_data.get("latex"))
                    break
        print(f"Gemini Retrieval Plan: Concepts: {retrieved_concept_names}, Equations: {retrieved_equations_latex}, Given: {given_values_parsed}, Target: {target_unknown_parsed}")

    except Exception as e:
        print(f"Error in Gemini retrieval planning: {e}")
        return SolvePhysicsProblemOutput(
            original_question=question,
            identified_concepts=retrieved_concept_names,
            relevant_equations_latex=retrieved_equations_latex,
            explanation=f"Error during problem analysis: {e}",
            status="error_analysis"
        )

    # --- Stage 2: Symbolic Solution with SymPy (Conceptual & Highly Simplified) ---
    symbolic_steps_str = "Symbolic solver stage: Not fully implemented for general problems."
    final_answer_str = None

    if retrieved_equations_latex:
        # This is where the extremely challenging part of general symbolic solving would go.
        # It requires:
        # 1. Parsing LaTeX into SymPy expressions.
        # 2. Parsing `given_values_parsed` and `target_unknown_parsed` to create SymPy symbols and knowns.
        # 3. Setting up a system of equations.
        # 4. Solving for the target unknown.
        # This is beyond a simple example for a general physics solver.
        # We'll make a placeholder attempt if specific equations are found.

        # Example: If F = m*a is retrieved, and question gives m and a.
        # This is hard to generalize without a robust NLP-to-SymPy layer.
        symbolic_steps_str = "Symbolic solving attempted with retrieved equations:\n"
        try:
            # Placeholder for a more advanced solver
            if any("F = m * a" in eq.replace(" ", "") for eq in retrieved_equations_latex) and \
               any(gv.get("variable_guess") == "m" for gv in given_values_parsed) and \
               any(gv.get("variable_guess") == "a" for gv in given_values_parsed) and \
               "F" in target_unknown_parsed:

                m_val = next((gv["value"] for gv in given_values_parsed if gv.get("variable_guess") == "m"), None)
                a_val = next((gv["value"] for gv in given_values_parsed if gv.get("variable_guess") == "a"), None)
                
                if m_val is not None and a_val is not None:
                    F, m, a = sympy.symbols('F m a')
                    eq = sympy.Eq(F, m * a)
                    # Convert string values to numbers
                    m_val_num = float(m_val)
                    a_val_num = float(a_val)
                    solution = sympy.solve(eq.subs({m: m_val_num, a: a_val_num}), F)
                    if solution:
                        final_answer_str = f"{F} = {solution[0]}"
                        symbolic_steps_str += f"Using F = m * a. Substituting m={m_val_num}, a={a_val_num}. Solved for F: {solution[0]}."
                    else:
                        symbolic_steps_str += "Could not solve F = m * a with provided values."
                else:
                    symbolic_steps_str += "F = m * a retrieved, but mass or acceleration values not identified from question by Gemini."
            else:
                symbolic_steps_str += "No direct rule-based symbolic solution applied for this combination."
            
            if not final_answer_str and retrieved_equations_latex:
                 symbolic_steps_str += "\nFurther symbolic manipulation would be required for a general solution."


        except Exception as e:
            print(f"Error during simplified symbolic attempt: {e}")
            symbolic_steps_str += f"\nError in symbolic processing: {e}"
    else:
        symbolic_steps_str = "No specific equations were identified by Gemini for symbolic solution."
    
    print(f"Symbolic Solution (Conceptual): {symbolic_steps_str}")
    if final_answer_str: print(f"Final Answer (Conceptual): {final_answer_str}")


    # --- Stage 3: Gemini for Explanation and Solution Synthesis ---
    # This stage will use the analyzed information to generate a coherent answer.

    prompt_solve_and_explain = f"""
    You are a physics problem solver and explainer.

    Original Physics Question: "{question}"

    Identified Key Concepts:
    {', '.join(retrieved_concept_names) if retrieved_concept_names else "None specified by analysis."}

    Relevant Equations (LaTeX) identified for potential use:
    {', '.join(retrieved_equations_latex) if retrieved_equations_latex else "None specified by analysis."}

    Values given in the question (as interpreted):
    {json.dumps(given_values_parsed, indent=2) if given_values_parsed else "None explicitly parsed."}

    Target to solve for (as interpreted):
    {target_unknown_parsed}

    Conceptual Symbolic Solution Steps/Outcome:
    {symbolic_steps_str}

    Concise Final Answer (if calculated):
    {final_answer_str if final_answer_str else "Not explicitly calculated by the symbolic stage."}

    Based on ALL the above information, provide a comprehensive, step-by-step explanation of how to approach and solve the original question.
    - Clearly state the principles involved.
    - Explain which equations are relevant and why.
    - Describe the steps to solve the problem. If a symbolic solution was attempted, integrate its outcome.
    - If the problem cannot be solved with the given information or retrieved knowledge, explain what is missing or why.
    - Be precise, pedagogical, and use clear physics terminology.
    - If a numerical answer was derived, present it clearly.

    Begin the explanation:
    """
    explanation_text = "Could not generate a full explanation."
    solution_synthesis_duration = 0

    try:
        print("Asking Gemini for final solution and explanation...")
        solution_synthesis_start_time = time.time()
        response_solve = await client.aio.models.generate_content(model=gemini_model.name, contents=prompt_solve_and_explain)
        solution_synthesis_end_time = time.time()
        solution_synthesis_duration = solution_synthesis_end_time - solution_synthesis_start_time
        print(f"DEBUG: Gemini solution synthesis call took {solution_synthesis_duration:.2f} seconds.")
        
        # Clean potential markdown and extract JSON (if expecting JSON)
        # For now, let's assume it's mostly text, but could be structured.
        explanation_text = response_solve.text.strip()

    except Exception as e:
        print(f"Error in Gemini solution synthesis: {e}")
        # Populate with what we have, even if synthesis fails
        return SolvePhysicsProblemOutput(
            original_question=question,
            identified_concepts=retrieved_concept_names,
            relevant_equations_latex=retrieved_equations_latex,
            symbolic_solution_steps=symbolic_steps_str,
            final_answer=final_answer_str,
            explanation="Error during explanation generation.",
            status="error_explanation"
        )

    print(f"DEBUG: Total time spent in Gemini calls: {retrieval_duration + solution_synthesis_duration:.2f} seconds.")
    return SolvePhysicsProblemOutput(
        original_question=question,
        identified_concepts=retrieved_concept_names,
        relevant_equations_latex=retrieved_equations_latex,
        symbolic_solution_steps=symbolic_steps_str,
        final_answer=final_answer_str,
        explanation=explanation_text,
        status="explanation_provided"
    )

# In the future, you will add your MCP tool and resource handlers here.
# For example:
# @app.tool()
# def solve_physics_problem(query: str) -> str:
# """Solves a physics problem based on the provided query."""
#     # ... implementation using knowledge base, LLM, and SymPy ...
#     return "Solution to " + query


if __name__ == "__main__":
    print("Starting Physics Knowledge MCP Server...")
    # This runs the server, making it accessible via stdio by default.
    # Clients like MCP Inspector or Claude Desktop can then connect.
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"Error running MCP server: {e}")
    finally:
        print("Physics Knowledge MCP Server stopped.")