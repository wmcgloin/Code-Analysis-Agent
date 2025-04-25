"""
LangGraph RAG System for Code Repositories

This module defines a Retrieval-Augmented Generation (RAG) system tailored for software repositories.
It includes:
- `RAGVectorDBSetup`: Uses an LLM to generate natural language descriptions of code files, then stores them in a vector store.
- `RAGQueryEngine`: Enables semantic querying using LangGraph and those descriptions.
- `create_rag_system`: A one-step initializer to build both components.

The goal is to support intelligent, context-aware queries over codebases.
"""

import os
from typing import Dict, List, Optional

from typing_extensions import Annotated, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model

from utils.repo import generate_repo_tree


class State(TypedDict):
    """LangGraph application state for semantic code queries."""
    question: str
    context: List[Document]
    answer: str


class RAGVectorDBSetup:
    """
    Extracts code semantics and builds a vector store of natural language representations.

    This class:
    - Generates a tree overview of the repo
    - Produces detailed file-level descriptions using an LLM
    - Stores those descriptions in an in-memory vector DB
    """

    def __init__(self, repo_path: str, embedding_model: str = "text-embedding-3-large"):
        """
        Initialize the vector database setup.

        Args:
            repo_path: Path to the target code repository.
            embedding_model: Name of the OpenAI embedding model to use.
        """
        self.repo_path = repo_path
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.llm = init_chat_model("gpt-4o", model_provider="openai")

        self.code_description_prompt = """
        You are an expert in translating code into natural language.
        Your goal is to generate a natural language description of the code in the given file.
        The description should be in a natural language format, and in great detail, but it should not be too verbose.
        """

        self.repo_tree = None
        self.descriptions: Dict[str, str] = {}

    def generate_code_descriptions(self) -> Dict[str, str]:
        """
        Traverse the repo and use an LLM to generate natural language descriptions for each code file.

        Returns:
            Dictionary mapping file paths to LLM-generated descriptions.
        """
        descriptions = {}

        if not self.repo_tree:
            self.repo_tree = generate_repo_tree(self.repo_path)

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for file in files:
                if file.startswith("."):
                    continue
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                rel_path = relative_path.replace("\\", ".")  # Normalize for module names

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()

                    messages = [
                        SystemMessage(self.code_description_prompt),
                        HumanMessage(f"""
                            Tree:
                            {self.repo_tree}

                            Current File Path:
                            {rel_path}

                            Code:
                            {code}
                        """)
                    ]
                    response = self.llm.invoke(messages)
                    descriptions[rel_path] = response.content

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        self.descriptions = {k.replace('\\', '.'): v for k, v in descriptions.items()}
        return self.descriptions

    def build_vector_store(self):
        """
        Split natural language descriptions and store them in a vector database.
        
        Returns:
            An InMemoryVectorStore containing the embedded document chunks.
        """
        if not self.descriptions:
            self.generate_code_descriptions()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        all_docs = []
        for file_path, description in self.descriptions.items():
            docs = splitter.create_documents([description])
            for doc in docs:
                doc.metadata["source"] = file_path
            all_docs.extend(docs)

        self.vector_store.add_documents(all_docs)
        return self.vector_store

    def setup_vector_database(self):
        """
        Generate descriptions and populate the vector database.

        Returns:
            The completed vector store.
        """
        self.generate_code_descriptions()
        self.build_vector_store()
        return self.vector_store


class RAGQueryEngine:
    """
    Executes LangGraph-based semantic search over code descriptions.
    """

    def __init__(self, vector_store: InMemoryVectorStore, llm_model: str = "gpt-4o"):
        """
        Initialize the query engine.

        Args:
            vector_store: A vector DB with embedded code descriptions.
            llm_model: The LLM model to use for final synthesis.
        """
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.llm = init_chat_model(llm_model, model_provider="openai")

        self.rag_system_prompt = """
        You are an expert in adding relevant details to a GRAPH RAG queried statement. You will be given a sentence to analyze (which is the output of a cypher query) and a context (which is additional information on the nodes and relationships). 
        Your task is to add relevant details to the sentence to make it more informative and useful. The context will be used to provide additional information about the entities and relationships in the sentence.
        Sentence to analyze: {question}
        Context: {context}
        Answer:
        """

        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_system_prompt)
        self.graph = None

    def _retrieve(self, state: State):
        """Retrieve top-K semantically similar documents from the vector DB."""
        return {"context": self.vector_store.similarity_search(state["question"], k=10)}

    def _generate(self, state: State):
        """Generate an enriched response using the question and retrieved context."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.rag_prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def build_graph(self):
        """
        Build and compile a LangGraph pipeline consisting of:
        - Retrieval
        - Generation
        """
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        self.graph = graph_builder.compile()
        return self.graph

    def query(self, question: str) -> str:
        """
        Query the RAG system.

        Args:
            question: A natural language prompt related to the code.

        Returns:
            The LLM-generated enriched answer.
        """
        if not self.graph:
            self.build_graph()
        return self.graph.invoke({"question": question})["answer"]


def create_rag_system(
    repo_path: str,
    embedding_model: str = "text-embedding-3-large",
    llm_model: str = "gpt-40",
) -> tuple[RAGVectorDBSetup, RAGQueryEngine]:
    """
    Orchestrates end-to-end RAG system initialization from a code repository.

    Args:
        repo_path: Path to the code repository.
        embedding_model: Embedding model to use.
        llm_model: Chat model to use for retrieval synthesis.

    Returns:
        Tuple of (vector DB setup, query engine)
    """
    # Initialize the vector database setup
    db_setup = RAGVectorDBSetup(repo_path, embedding_model)
    vector_store = db_setup.setup_vector_database()

    # Initialize the query engine
    query_engine = RAGQueryEngine(vector_store, llm_model)
    
    # Build the query graph
    query_engine.build_graph()

    return db_setup, query_engine