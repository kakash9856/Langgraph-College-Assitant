
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Load API key
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=api_key
)
# 1. Define State
class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


# 2. Editor Node
def editor_node(state: PipelineState) -> dict:
    """Stage 1: Correct grammar and spelling."""

    print("\n--- Stage 1: Editor Node ---")

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix grammatical errors and spelling mistakes while keeping "
        "the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {"edited_text": response.content.strip()}


# 3. Scriptwriter Node
def scriptwriter_node(state: PipelineState) -> dict:
    """Stage 2: Convert text into an engaging video script."""

    print("\n--- Stage 2: Scriptwriter Node ---")

    prompt = (
        "You are a charismatic YouTube content creator. Transform "
        "the following edited text into an engaging and conversational "
        "video script. Return only the script.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {"script_text": response.content.strip()}


# 4. Translator Node
def translator_node(state: PipelineState) -> dict:
    """Stage 3: Translate script into natural Hinglish."""

    print("\n--- Stage 3: Translator Node ---")

    prompt = (
        "Convert the following script into natural, flowing Hinglish. "
        "Use a comfortable mix of Hindi and English, like an Indian "
        "tech educator speaking naturally. Return only the final text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {"final_output": response.content.strip()}


# 5. Create Graph
graph = StateGraph(PipelineState)

# Add Nodes
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

# Add Edges
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)

# 6. Compile Graph
app = graph.compile()

# 7. Invoke Graph
result = app.invoke({
    "raw_input": (
        "AI agents are the future of tech. They can think, plan, "
        "and act on their own. LangGraph helps you build these "
        "agents with proper control and memory."
    )
})

# 8. Final Output
print("\nYour Final Hinglish Output:\n")
print(result["final_output"])
