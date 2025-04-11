"""
RAG Graph module - Creates a LangGraph-based RAG system 
using code repository descriptions.
"""
import os
from typing import Dict, List, Optional

from typing_extensions import Annotated, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph


class State(TypedDict):
    """State for the RAG graph application."""
    question: str
    context: List[Document]
    answer: str


class RAGVectorDBSetup:
    """A class that builds and manages the vector database for code repositories."""
    
    def __init__(
        self, 
        repo_path: str,
        embedding_model: str = "text-embedding-3-large",
    ):
        """
        Initialize the RAG Vector Database Setup.
        
        Args:
            repo_path: Path to the code repository
            embedding_model: Model to use for embeddings
        """
        self.repo_path = repo_path
        self.embedding_model = embedding_model
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        
        # Setup LLM for code description
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
        
        # Setup LLM prompts
        self.code_description_prompt = """
        You are an expert in translating code into natural language.
        Your goal is to generate a natural language description of the code in the given file.
        The description should be in a natural language format, and in great detail, but it should not be too verbose.
        """
        
        # Initialize repo data
        self.repo_tree = None
        self.descriptions = {}
        
    def generate_repo_tree(self) -> str:
        """Generate a string representation of the repository structure."""
        tree_string = ""
        for root, dirs, files in os.walk(self.repo_path):
            # Filter out __pycache__ and hidden directories
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]

            level = root.replace(self.repo_path, "").count(os.sep)
            indent = "│   " * level + "├── "  # Formatting the tree
            tree_string += f"{indent}{os.path.basename(root)}/\n"

            sub_indent = "│   " * (level + 1) + "├── "
            for file in files:
                tree_string += f"{sub_indent}{file}\n"

        self.repo_tree = tree_string
        return tree_string
        
    def generate_code_descriptions(self) -> Dict[str, str]:
        """Generate natural language descriptions for all code files in the repository."""
        descriptions = {}
        
        if not self.repo_tree:
            self.generate_repo_tree()
            
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for file in files:
                if file.startswith("."):
                    continue  # Skip hidden files

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                rel_path = relative_path.replace("\\", ".")
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()

                    messages = [
                        SystemMessage(self.code_description_prompt),
                        HumanMessage(f'''
                            Tree:
                            {self.repo_tree}

                            Current File Path:
                            {rel_path}

                            Code:
                            {code}
                        ''')
                    ]

                    response = self.llm.invoke(messages)
                    descriptions[rel_path] = response.content

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # Normalize keys
        descriptions = {k.replace('\\', '.'): v for k, v in descriptions.items()}
        self.descriptions = descriptions
        return descriptions
        
    def build_vector_store(self):
        """Build the vector store from code descriptions."""
        if not self.descriptions:
            self.generate_code_descriptions()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        all_docs = []
        for file_path in self.descriptions:
            docs = text_splitter.create_documents([self.descriptions[file_path]])
            for doc in docs:
                doc.metadata['source'] = file_path
            all_docs.extend(docs)

        self.vector_store.add_documents(documents=all_docs)
        return self.vector_store

    def setup_vector_database(self):
        """Complete setup of the vector database."""
        self.generate_repo_tree()
        self.generate_code_descriptions()
        self.build_vector_store()
        return self.vector_store


class RAGQueryEngine:
    """A class that handles querying the RAG system."""
    
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        llm_model: str = "claude-3-5-sonnet-20240620",
    ):
        """
        Initialize the RAG Query Engine.
        
        Args:
            vector_store: The initialized vector store
            llm_model: Model to use for LLM
        """
        self.vector_store = vector_store
        self.llm_model = llm_model
        
        # Initialize LLM
        self.llm = ChatAnthropic(model=llm_model)
        
        # Setup RAG prompt
        self.rag_system_prompt = """
        You are an expert in adding relevant details to a GRAPH RAG queried statement. You will be given a sentence to analyze (which is the output of a cypher query) and a context (which is additional information on the nodes and relationships). 
        Your task is to add relevant details to the sentence to make it more informative and useful. The context will be used to provide additional information about the entities and relationships in the sentence.
        Sentence to analyze: {question}
        Context: {context}
        Answer:
        """
        
        self.rag_prompt = ChatPromptTemplate.from_template(self.rag_system_prompt)
        
        # Initialize the graph
        self.graph = None
    
    def _retrieve(self, state: State):
        """Retrieve relevant documents based on the question."""
        retrieved_docs = self.vector_store.similarity_search(state["question"], k=10)
        return {"context": retrieved_docs}

    def _generate(self, state: State):
        """Generate an answer based on the retrieved documents."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.rag_prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}
        
    def build_graph(self):
        """Build the LangGraph for RAG."""
        # Create the graph
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        self.graph = graph_builder.compile()
        return self.graph
        
    def query(self, question: str) -> str:
        """Query the RAG graph with a question."""
        if not self.graph:
            self.build_graph()
            
        result = self.graph.invoke({"question": question})
        return result["answer"]


# Function to create and initialize the RAG system
def create_rag_system(
    repo_path: str,
    embedding_model: str = "text-embedding-3-large",
    llm_model: str = "claude-3-5-sonnet-20240620",
) -> tuple[RAGVectorDBSetup, RAGQueryEngine]:
    """
    Create and initialize the RAG system for the given repository.
    
    Args:
        repo_path: Path to the code repository
        embedding_model: Model to use for embeddings
        llm_model: Model to use for LLM
        
    Returns:
        A tuple containing the RAGVectorDBSetup and RAGQueryEngine objects
    """
    # Create and set up the vector database
    db_setup = RAGVectorDBSetup(
        repo_path=repo_path,
        embedding_model=embedding_model,
    )
    
    vector_store = db_setup.setup_vector_database()
    
    # Create the query engine
    query_engine = RAGQueryEngine(
        vector_store=vector_store,
        llm_model=llm_model,
    )
    
    # Build the query graph
    query_engine.build_graph()
    
    return db_setup, query_engine


