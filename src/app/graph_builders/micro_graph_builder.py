"""
Micro-Level Code Graph Builder

This module defines the `MicroCodeGraphBuilder` class, which uses a large language model
to convert Python code files into a structured graph of semantic relationships.
It performs the following:

- Generates detailed natural language descriptions of classes, functions, and their connections.
- Normalizes descriptions to ensure consistent node identifiers for graph construction.
- Converts these descriptions into graph documents compatible with Neo4j.
- Uploads the resulting graph to a Neo4j database.

This is part of the micro-level processing pipeline and is triggered during setup.
"""

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

from utils import get_logger
from utils.filesystem import read_code_file, list_python_files
from utils.repo import generate_repo_tree

import asyncio
from langchain_core.messages import HumanMessage, SystemMessage

import random

# Load environment variables from .env file (for Neo4j credentials)
load_dotenv()

logger = get_logger()


class MicroCodeGraphBuilder:
    """
    Class responsible for generating a semantic code graph from a Python repository.

    This includes:
    - Extracting semantic relationships using an LLM
    - Normalizing and refining relationships for graph structure
    - Converting descriptions into graph documents
    - Uploading the final graph to a Neo4j database
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the MicroCodeGraphBuilder.

        Args:
            llm: The language model to use for code understanding
        """
        self.llm = llm

        # Prompt for initial code analysis
        self.system_prompt = """
        You are an expert in analyzing Python code and generating structured natural language descriptions for graph-based querying in Cypher. 
        Given a Python codebase, extract meaningful relationships between functions, classes, and imported modules. 
        
        Only use the list of types provided below:
        - class : classes defined in a module
        - method : methods defined in a class
        - function : functions defined in a module
        - module : python scripts defined within the repository. Exclude .py when mentioning the module name.
        - package : packages imported that are not modules.
        Do not include information on variables, parameters, arguments.
        Python scripts must be modules and pre defined packages such as numpy and pandas must be packages

        When generating the structured natural language description, follow these rules:    
        - Do not give explanations for the code logic or functionality.
        - Do not use adjectives and adverbs. 
        - Only describe the code and do not give an overall summary.
        - Do not use ambiguous pronouns and use exact names in every description.
        - Explain each class, function separately and do not include explanations such as 'as mentioned before' or anything that refers to a previous explanation.
        - make each description sufficient for a standalone statement for one relationship in the graph.    
        - Each class and funciton should be connected to the module where it was defined.
        - Each imported package should be connected to the function, method or class where it was used.
        - Always include an explanation on how the outermost class or method is connected to the module where it is defined.
        - If the outermost layer is an 'if __name__ == "__main__":' block, then the outermost layer is whatever is inside the block. Plus whatever is defined outside the block. Make sure to mention the connection between the module and the closses and functions.
        - When mentioning modules, take note of the current file path(relative repository) given in input, and change slashes to dots and remove the .py extension.
        - If a function or class is used in another function or class, make sure to mention the connection between them.
        - 
                
        Natural language should follow a similar format as below:
            {source.id} is a {source.type} with properties {source.properties} defined in {target.id} which is a {target.type}.        
        Example: 
        - When mentioning classes, always refer them as {relative_repository}.{module_name}.{class_name}
        - When mentioning methods, always refer to them as {relative_repository}.{module_name}.{class_name}.{method_name}
        - When mentioning functions, always refer them as {relative_repository}.{module_name}.{function_name}
        - If the file path is deduplication/LSH.py and there is a class LSH in it, the module is deduplication.LSH and the class is deduplication.LSH.LSH.

        Example:
        deduplication.LSHImproved.LSHImproved is a module that defines the class deduplication.LSH.lsh_base, which consists of  method deduplication.LSH.lsh_base.hash_function.
        deduplication.LSH.lsh_base is a class and inherits from the class utils.utils.BaseLSH.
        numpy is a package and is used in the method deduplication.LSH.lsh_base.hash_function.
        bitarray is a package and is used in the deduplication.bloom_filter.BloomFilter_KM_Opt.__init__ method of the class deduplication.bloom_filter.BloomFilter_KM_Opt.  
        deduplication.LSH.lsh_base is a class defined in the module deduplication.LSH.
        
        If a module from our repository is imported in another module in our repository, refer to it as the entire path of the module.
        Example:
        code within deduplication\\__main__.py : from deduplication.LSHImproved import LSHImproved
        Natural Language Description:
        The Class deduplication.LSHImproved.LSHImproved is imported into the module deduplication.__main__.
        """

        # Prompt for refinement of node paths
        self.refine_prompt = """
        You are an expert text editor. Your goal is to modyfy the text in a way that it is consistent with the given repository tree and file path.
        Keep in mind this natural language is meant to be used for graph-based querying in Cypher.
        Only output the modified text, do not give any explanations or anything else.
        The only change you have to make is to modify potential node ids, so they are consistent and can be connected by a graph.
        
        Rules:
        - When mentioning modules, take note of the current file path(relative repository) given in input, and change slashes to dots and remove the .py extension.
        - This means that the current file path or module name should be the beginning of all classes/methods/functions defined in the module.    
        - When mentioning classes, always refer them as {module_location_name}.{class_name}
        - When mentioning methods, always refer to them as {module_location_name}.{class_name}.{method_name}
        - When mentioning functions, always refer them as {module_location_name}.{function_name}

        Example: 
        - if the relative path is deduplication/LSH.py and there is a class LSH in it, the module is deduplication.LSH and the class is deduplication.LSH.LSH.
        - Given the current file path deduplication\\dedup.py
            'deduplication.Baseline is a class defined in deduplication.dedup' is extremely incorrect. 
            The correct description is 'deduplication.dedup.Baseline is a class defined in deduplication.dedup'
        
        
        Example:
        - If the file location is deduplication/LSH.py and there is a class LSH in it, the module should be deduplication.LSH and the class should be deduplication.LSH.LSH. in the natural language description.
        - If the original text is bloom_filter.BloomFilter is a class, and the current file is deduplication\\bloom_filter.py, the modified text should be deduplication.bloom_filter.BloomFilter is a class.
        - If my current module imports a function from a module 'utils.utils import clean_document' the modified text should be utils.utils.clean_document is a function.
        """

        # LLM to Graph Transformer setup
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["class", "method", "function", "package", "module"],
        )

    async def build_graph_and_upload(self, repo_path: str) -> List[Dict[str, Any]]:
        """Analyze a Python codebase and upload its semantic graph to Neo4j."""
        repo_tree_string = generate_repo_tree(repo_path)
        descriptions = {}

        # Step 1: Generate natural language descriptions of code using async
        python_files = list_python_files(repo_path)
        
        async def process_file(relative_path):
            if not relative_path.endswith('.py'):
                return  # Skip non-python files early
            print(f"Processing file parallel: {relative_path}")
            file_path = os.path.join(repo_path, relative_path)
            try:
                code_content = read_code_file(file_path)
                if code_content is None:
                    return  # Skip unreadable file
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"""
                        Tree:
                        {repo_tree_string}

                        Current File Path:
                        {relative_path}

                        Code:
                        {code_content}
                    """),
                ]
                # Use the async version of invoke
                response = await self.llm.ainvoke(messages)
                return (relative_path, response.content)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                return None
        
        # Run all file processing concurrently
        tasks = [process_file(relative_path) for relative_path in python_files]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and convert to dictionary
        descriptions = {rel_path: content for result in results if result is not None for rel_path, content in [result]}
        
        # Similarly, you can parallelize the refinement step
        async def refine_description(file_path, description):
            messages = [
                SystemMessage(content=self.refine_prompt),
                HumanMessage(content=f"""
                    Repository Tree:
                    {repo_tree_string}

                    Current File Path:
                    {file_path}

                    Natural Language Description:
                    {description}
                """),
            ]
            response = await self.llm.ainvoke(messages)
            return (file_path, response.content)
        
        # Run all refinement tasks concurrently
        refine_tasks = [refine_description(file_path, description) 
                    for file_path, description in descriptions.items()]
        refine_results = await asyncio.gather(*refine_tasks)
        refined_descriptions = dict(refine_results)








        # Step 3: Convert to LangChain-compatible Document objects
        description_documents = [
            Document(page_content=desc, metadata={"source": file_path})
            for file_path, desc in refined_descriptions.items()
        ]

        # Step 4: Transform documents into graph nodes/edges in parallel
        print("Converting to graph documents in parallel...")

        async def process_document(document):
            """Process a single document to generate graph documents"""
            # Create a document-specific transformer to avoid concurrency issues
            # If this is resource-intensive, you might want to reuse the transformer
            transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=["class", "method", "function", "package", "module"],
            )
            
            # Process single document - this depends on the LLMGraphTransformer implementation
            # If convert_to_graph_documents doesn't have a single-document version, you might need to modify it
            try:
                # Add rate limiting to avoid hitting API limits
                await asyncio.sleep(random.uniform(0.2, 0.5))
                result = transformer.convert_to_graph_documents([document])
                return result
            except Exception as e:
                print(f"Error processing document {document.metadata.get('source', 'unknown')}: {e}")
                return []

        # Create tasks for all documents
        # document_tasks = [process_document(doc) for doc in description_documents]

        # Use semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(5)  # Process 5 documents at a time
        async def bounded_process_document(document):
            async with semaphore:
                return await process_document(document)

        # Process documents with concurrency control
        document_results = await asyncio.gather(*[bounded_process_document(doc) for doc in description_documents])

        # Flatten results list
        graph_documents = []
        for result in document_results:
            if result:
                graph_documents.extend(result)

    # def build_graph_and_upload(self, repo_path: str) -> List[Dict[str, Any]]:
    #     """
    #     Analyze a Python codebase and upload its semantic graph to Neo4j.

    #     Args:
    #         repo_path: Path to the local repository directory

    #     Returns:
    #         A tuple of:
    #         - Neo4jGraph instance with schema loaded
    #         - List of graph documents created from the codebase
    #     """
    #     from langchain_core.messages import HumanMessage, SystemMessage

    #     repo_tree_string = generate_repo_tree(repo_path)
    #     descriptions = {}

    #     # Step 1: Generate natural language descriptions of code
    #     python_files = list_python_files(repo_path)
    #     for relative_path in python_files:
    #         if not relative_path.endswith('.py'):
    #             continue  # Skip non-python files early
    #         print(f"Processing file: {relative_path}")
    #         file_path = os.path.join(repo_path, relative_path)
    #         try:
    #             code_content = read_code_file(file_path)
    #             if code_content is None:
    #                 continue  # Skip unreadable file
    #             messages = [
    #                 SystemMessage(content=self.system_prompt),
    #                 HumanMessage(content=f"""
    #                     Tree:
    #                     {repo_tree_string}

    #                     Current File Path:
    #                     {relative_path}

    #                     Code:
    #                     {code_content}
    #                 """),
    #             ]
    #             response = self.llm.invoke(messages)
    #             descriptions[relative_path] = response.content
    #         except Exception as e:
    #             print(f"Error processing {file_path}: {e}")

    #     # Step 2: Refine descriptions for graph-consistent node naming
    #     refined_descriptions = {}
    #     for file_path, description in descriptions.items():
    #         messages = [
    #             SystemMessage(content=self.refine_prompt),
    #             HumanMessage(content=f"""
    #                 Repository Tree:
    #                 {repo_tree_string}

    #                 Current File Path:
    #                 {file_path}

    #                 Natural Language Description:
    #                 {description}
    #             """),
    #         ]
    #         response = self.llm.invoke(messages)
    #         refined_descriptions[file_path] = response.content

        # # Step 3: Convert to LangChain-compatible Document objects
        # description_documents = [
        #     Document(page_content=desc, metadata={"source": file_path})
        #     for file_path, desc in refined_descriptions.items()
        # ]

        # # Step 4: Transform documents into graph nodes/edges
        # print("Converting to graph documents...")
        # graph_documents = self.llm_transformer.convert_to_graph_documents(description_documents)

        # Step 5: Upload to Neo4j
        try:
            url = os.getenv('NEO4J_URI')
            username = os.getenv('NEO4J_USERNAME')
            password = os.getenv('NEO4J_PASSWORD')
            graph = Neo4jGraph(url=url, username=username, password=password, enhanced_schema=True)

            # Clear existing nodes/edges before uploading new graph
            graph.query("MATCH (n) DETACH DELETE n")
            print("Cleared existing graph data in Neo4j.")
        except Exception as e:
            return print(f"Error connecting to Neo4j: {e}")

        print("Adding graph documents to Neo4j...")
        graph.add_graph_documents(graph_documents)
        graph.refresh_schema()

        return graph, graph_documents
    




