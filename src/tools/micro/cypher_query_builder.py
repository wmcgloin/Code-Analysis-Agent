from operator import add
from typing import Annotated, List, Literal, Optional
import logging

from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_neo4j import Neo4jVector
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from neo4j.exceptions import CypherSyntaxError
from langgraph.graph import END, START, StateGraph
from langchain_openai import OpenAIEmbeddings

# Define TypedDict classes for state management
class InputState(TypedDict):
    question: str

class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]

class OutputState(TypedDict):
    answer: str
    steps: List[str]
    cypher_statement: str
    cypher_results: List[str]

# Define Pydantic models for validation
class Property(BaseModel):
    """
    Represents a filter condition based on a specific node property in a graph in a Cypher statement.
    """
    node_label: str = Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(description="The key of the property being filtered.")
    property_value: str = Field(
        description="The value that the property is being matched against."
    )

class ValidateCypherOutput(BaseModel):
    """
    Represents the validation result of a Cypher query's output,
    including any errors and applied filters.
    """
    errors: Optional[List[str]] = Field(
        description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
    )
    filters: Optional[List[Property]] = Field(
        description="A list of property-based filters applied in the Cypher statement."
    )


# MATCH path = (m1:Module {id: 'Utils.Utils'})-[*..5]-(m2:Module {id: 'Deduplication.__Main__'}) WHERE NONE(n IN nodes(path) WHERE n:Package) RETURN [n IN nodes(path) | labels(n)] AS nodeTypes, path LIMIT 20
# Default examples for few-shot learning
default_examples = [
    {
        "question": "How is the Utils.Utils module related to the Deduplication.__Main__ module?",
        "query": "MATCH path = (m1:Module {id: 'Utils.Utils'})-[*..5]-(m2:Module {id: 'Deduplication.__Main__'}) WHERE NONE(n IN nodes(path) WHERE n:Package) RETURN [n IN nodes(path) | labels(n)] AS nodeTypes, path LIMIT 20",
    },
    {
        "question": "Where is the Utils.Utils model imported?",
        "query": "MATCH path = (m:Module {id: 'Utils.Utils'})<-[r]-(importer:Module) RETURN [n IN nodes(path) | labels(n)] AS nodeTypes, path LIMIT 20",
    },
    {
        "question": "What functions make up the Utils.Utils.Utilities class?",
        "query": "MATCH path = (c:Class {id: 'Utils.Utils.Utilities'})-[]-(f:Function) RETURN [n IN nodes(path) | labels(n)] AS nodeTypes, path LIMIT 20",
    },
    {
        "question": "What classes are connected to the LSH.LSHForest module?",
        "query": "MATCH path = (m:Module {id: 'LSH.LSHForest'})-[*..5]-(c:Class) RETURN [n IN nodes(path) | labels(n)] AS nodeTypes, path LIMIT 20",
    },
    {
        "question": "How is the Src.App.Handlers.Query_Handler module related to the Src.Streamlit_App module?",
        "query": "MATCH path = (m1:Module {id: 'Src.App.Handlers.Query_Handler'})-[*..5]-(m2:Module {id: 'Src.Streamlit_App'}) WHERE NONE(n IN nodes(path) WHERE n:Package) RETURN [n IN nodes(path) | labels(n)] AS nodeTypes, path LIMIT 20",
    },
]

def create_example_selector(examples, k=5):
    """Create an example selector for few-shot prompting"""
    return SemanticSimilarityExampleSelector.from_examples(
        examples, OpenAIEmbeddings(), Neo4jVector, k=k, input_keys=["question"]
    )

def create_text2cypher_chain(llm):
    """Create a chain for converting text to Cypher queries"""
    text2cypher_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
                "Do not change any node ids and labels given by the user."
                "Always start all Cypher statements with 'MATCH path = ' to find multiple paths."
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
                Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
                Here is the schema information
                {schema}
                
                Below are a number of examples of questions and their corresponding Cypher queries.
                {fewshot_examples}
                
                User input: {question}
                Cypher query:"""
            ),
        ),
    ])
    return text2cypher_prompt | llm | StrOutputParser()

def create_validate_cypher_chain(llm):
    """Create a chain for validating Cypher queries"""
    validate_cypher_system = """
    You are a Cypher expert reviewing a statement written by a junior developer.
    """

    validate_cypher_user = """You must check the following:
    * Are there any syntax errors in the Cypher statement?
    * Are there any missing or undefined variables in the Cypher statement?
    * Are any node labels missing from the schema?
    * Are any relationship types missing from the schema?
    * Are any of the properties not included in the schema?
    * Does the Cypher statement include enough information to answer the question?

    Examples of good errors:
    * Label (:Foo) does not exist, did you mean (:Bar)?
    * Property bar does not exist for label Foo, did you mean baz?
    * Relationship FOO does not exist, did you mean FOO_BAR?

    Schema:
    {schema}

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    Make sure you don't make any mistakes!"""

    validate_cypher_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            validate_cypher_system,
        ),
        (
            "human",
            (validate_cypher_user),
        ),
    ])

    return validate_cypher_prompt | llm.with_structured_output(ValidateCypherOutput)

def create_correct_cypher_chain(llm):
    """Create a chain for correcting Cypher queries"""
    correct_cypher_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """Check for invalid syntax or semantics and return a corrected Cypher statement.

    Schema:
    {schema}

    Note: Do not include any explanations or apologies in your responses.
    Do not wrap the response in any backticks or anything else.
    Respond with a Cypher statement only!

    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    The errors are:
    {errors}

    Corrected Cypher statement: """
            ),
        ),
    ])
    return correct_cypher_prompt | llm | StrOutputParser()

def create_generate_final_chain(llm):
    """Create a chain for generating the final answer"""
    generate_final_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """Use the following results retrieved from a database to provide
    a succinct, definitive answer to the user's question.

    Respond as if you are answering the question directly.
    One full connection in the graph is one answer.
    Answer up to 5 answers in bullet points if there are multiple connections.

    Results: {results}
    Question: {question}"""
            ),
        ),
    ])
    return generate_final_prompt | llm | StrOutputParser()

class CypherGraphBuilder:
    """Builder class for creating and managing a Cypher graph pipeline"""
    
    def __init__(self, llm, graph_db, examples=default_examples):
        """
        Initialize the Cypher Graph Builder
        
        Args:
            llm: Language model to use for generating and validating Cypher
            graph_db: Neo4j graph database connection
            examples: Optional list of examples for few-shot learning
        """
        self.llm = llm
        self.graph_db = graph_db
        self.examples = examples
        self.example_selector = create_example_selector(self.examples)
        
        # Initialize chains
        self.text2cypher_chain = create_text2cypher_chain(llm)
        self.validate_cypher_chain = create_validate_cypher_chain(llm)
        self.correct_cypher_chain = create_correct_cypher_chain(llm)
        self.generate_final_chain = create_generate_final_chain(llm)
        
        # Create schema for the Cypher query corrector
        self.create_corrector_schema()
        
    def create_corrector_schema(self):
        """Create schema for the Cypher query corrector"""
        structured_schema = self.graph_db.structured_schema
        relationships = structured_schema.get("relationships", [])
        self.corrector_schema = [Schema(r["start"], r["type"], r["end"]) for r in relationships]
        self.cypher_query_corrector = CypherQueryCorrector(self.corrector_schema)
    
    def generate_cypher(self, state: OverallState) -> OverallState:
        """
        Generates a cypher statement based on the provided schema and user input
        """
        NL = "\n"
        fewshot_examples = (NL * 2).join([
            f"Question: {el['question']}{NL}Cypher:{el['query']}"
            for el in self.example_selector.select_examples({"question": state.get("question")})
        ])
        
        generated_cypher = self.text2cypher_chain.invoke({
            "question": state.get("question"),
            "fewshot_examples": fewshot_examples,
            "schema": self.graph_db.schema,
        })
        
        logging.info(f"Generated Cypher: {generated_cypher}")
        return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}

    def validate_cypher(self, state: OverallState) -> OverallState:
        """
        Validates the Cypher statements and maps any property values to the database.
        """
        errors = []
        mapping_errors = []
        
        # Check for syntax errors
        try:
            self.graph_db.query(f"EXPLAIN {state.get('cypher_statement')}")
        except CypherSyntaxError as e:
            errors.append(e.message)
        
        # Use LLM to find additional potential errors and get the mapping for values
        llm_output = self.validate_cypher_chain.invoke({
            "question": state.get("question"),
            "schema": self.graph_db.schema,
            "cypher": state.get("cypher_statement"),
        })
        
        if llm_output.errors:
            errors.extend(llm_output.errors)
            
        if llm_output.filters:
            for filter in llm_output.filters:
                # Get node properties
                node_props = self.graph_db.structured_schema.get("node_props", {})
                node_label_props = node_props.get(filter.node_label, [])
                
                # Find property type
                prop_type = None
                for prop in node_label_props:
                    if prop["property"] == filter.property_key:
                        prop_type = prop["type"]
                        break
                
                # Do mapping only for string values
                if prop_type == "STRING":
                    mapping = self.graph_db.query(
                        f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                        {"value": filter.property_value},
                    )
                    if not mapping:
                        logging.info(
                            f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                        )
                        mapping_errors.append(
                            f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                        )
        
        if mapping_errors:
            next_action = "end"
        elif errors:
            next_action = "correct_cypher"
        else:
            next_action = "execute_cypher"

        return {
            "next_action": next_action,
            "cypher_errors": errors,
            "steps": ["validate_cypher"],
        }

    def correct_cypher(self, state: OverallState) -> OverallState:
        """
        Correct the Cypher statement based on the provided errors.
        """
        corrected_cypher = self.correct_cypher_chain.invoke({
            "question": state.get("question"),
            "errors": state.get("cypher_errors"),
            "cypher": state.get("cypher_statement"),
            "schema": self.graph_db.schema,
        })

        return {
            "next_action": "validate_cypher",
            "cypher_statement": corrected_cypher,
            "steps": ["correct_cypher"],
        }

    def execute_cypher(self, state: OverallState) -> OverallState:
        """
        Executes the given Cypher statement.
        """
        no_results = "I couldn't find any relevant information in the database"
        records = self.graph_db.query(state.get("cypher_statement"))
        
        return {
            "database_records": records if records else no_results,
            "next_action": "end",
            "steps": ["execute_cypher"],
        }

    def generate_final_answer(self, state: OverallState) -> OutputState:
        """
        Generates the final answer based on the database records.
        """
        final_answer = self.generate_final_chain.invoke({
            "question": state.get("question"), 
            "results": state.get("database_records")
        })
        
        logging.info(f"Database records: {state.get('database_records')}")
        
        return {
            "answer": final_answer, 
            "steps": ["generate_final_answer"], 
            'cypher_results': state.get('database_records'),
            'cypher_statement': state.get('cypher_statement')
        }

    def validate_cypher_condition(self, state: OverallState) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
        """
        Determines the next action based on the state.
        """
        if state.get("next_action") == "end":
            return "generate_final_answer"
        elif state.get("next_action") == "correct_cypher":
            return "correct_cypher"
        elif state.get("next_action") == "execute_cypher":
            return "execute_cypher"

    def create_cypher_graph(self):
        """
        Creates a state graph for the Cypher chain.
        """
        langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
        
        # Add nodes
        langgraph.add_node("generate_cypher", self.generate_cypher)
        # langgraph.add_node("validate_cypher", self.validate_cypher)
        # langgraph.add_node("correct_cypher", self.correct_cypher)
        langgraph.add_node("execute_cypher", self.execute_cypher)
        langgraph.add_node("generate_final_answer", self.generate_final_answer)

        # Add edges
        langgraph.add_edge(START, "generate_cypher")
        langgraph.add_edge("generate_cypher", "execute_cypher")
        # langgraph.add_conditional_edges("validate_cypher", self.validate_cypher_condition)
        langgraph.add_edge("execute_cypher", "generate_final_answer")
        # langgraph.add_edge("correct_cypher", "validate_cypher")
        langgraph.add_edge("generate_final_answer", END)

        return langgraph.compile()