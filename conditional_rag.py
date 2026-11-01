
import os
import streamlit as st
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

# ---------------- STREAMLIT CONFIG ----------------

st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

[data-testid="stSidebar"] {
    background-color: #151b27;
}

.main-title {
    font-size: 34px;
    font-weight: bold;
    color: white;
}

.subtitle {
    color: #aab4c5;
    font-size: 15px;
    margin-bottom: 20px;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD ENVIRONMENT ----------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()


# ---------------- STEP 1: BUILD RAG RETRIEVERS ----------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


@st.cache_resource
def build_retriever(pdf_path: str):

    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


try:
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")

except Exception as e:
    st.error(f"PDF loading error: {e}")
    st.stop()


# ---------------- GROQ MODEL ----------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4,
    api_key=api_key
)


# ---------------- STEP 2: STATE ----------------

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ---------------- STEP 3: NODES ----------------

def classifier_node(state: State) -> dict:

    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:

    query = state["messages"][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return {"retrieved_context": context}


def general_node(state: State) -> dict:

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


def response_node(state: State) -> dict:

    query = state["messages"][-1].content

    programme = state.get("programme", "Unknown")

    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )

    else:

        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)

    return {
        "messages": [("ai", response.content.strip())]
    }


# ---------------- STEP 4: ROUTER ----------------

def route_query(state: State):

    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    else:
        return "general"


# ---------------- STEP 5: BUILD GRAPH ----------------

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

graph.add_edge(START, "classifier")

graph.add_conditional_edges(
    "classifier",
    route_query
)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response", END)

app = graph.compile()


# ==================================================
#              STEP 6: STREAMLIT UI
# ==================================================


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "programme" not in st.session_state:
    st.session_state.programme = "BCA"


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## 🎓 College AI")

    st.caption("Your Personal Academic Assistant")

    st.divider()

    st.markdown("### 👨‍🎓 Student Profile")

    programme = st.selectbox(
        "Select your programme",
        ["BCA", "BBA", "B.Com (H)"],
        index=["BCA", "BBA", "B.Com (H)"].index(
            st.session_state.programme
        )
    )

    st.session_state.programme = programme

    st.success(f"Current Programme: {programme}")

    st.divider()

    if st.button("＋ New Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.markdown("### 💡 What I can help with")

    st.markdown("""
    📚 Academic Rules

    📝 Exams and Attendance

    🎯 Course Structure

    💰 Fee Structure

    💳 Payment and Refunds

    🎓 Degree Requirements
    """)

    st.divider()

    st.caption("Powered by LangGraph + RAG + Groq")


# ---------------- MAIN HEADER ----------------

st.markdown(
    '<div class="main-title">🎓 College AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent assistant for academic and fee-related queries.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- WELCOME SCREEN ----------------

if not st.session_state.messages:

    st.markdown("### 👋 Welcome to College AI Assistant!")

    st.write(
        f"I am your AI assistant. Ask me anything about your "
        f"**{st.session_state.programme}** programme."
    )

    st.markdown("### Try asking something")

    suggestions = [
        "What are the attendance requirements?",
        "Explain the examination rules.",
        "What is the fee refund policy?",
        "What are the degree requirements?"
    ]

    cols = st.columns(2)

    for i, question in enumerate(suggestions):

        with cols[i % 2]:

            if st.button(
                question,
                use_container_width=True
            ):

                st.session_state.pending_query = question
                st.rerun()


# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

user_query = st.chat_input(
    "Ask anything about academics, fees, exams..."
)

if "pending_query" in st.session_state:

    user_query = st.session_state.pop("pending_query")


# ---------------- PROCESS QUERY ----------------

if user_query:

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    # Prepare conversation history for LangGraph

    conversation = []

    for message in st.session_state.messages:

        role = "human" if message["role"] == "user" else "ai"

        conversation.append(
            (role, message["content"])
        )

    # Invoke LangGraph

    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking... Searching college documents..."):

            try:

                result = app.invoke({
                    "programme": st.session_state.programme,
                    "messages": conversation
                })

                answer = result["messages"][-1].content

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error("Something went wrong!")

                st.exception(e)

    st.rerun()
