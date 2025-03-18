# Milestone 2: Data Preparation and RAG Implementation

## 1. Overview
For our project, we’re building an AI-powered tool to help developers understand complex codebases more easily. Our system will generate logic flow diagrams and mind maps that provide better insights into how different parts of a repository interact. We’re using large language models (LLMs), retrieval-augmented generation (RAG), and graph databases to make this possible. The idea is to give developers an easy way to explore code structure, dependencies, and logic without having to dig through everything manually.

This milestone focused on preparing the data and getting our retrieval system working, so the model can find the most relevant pieces of code before generating explanations.

## 2. Data Preparation

### 2.1. Dataset Selection
To train and test our system, we’re using code from open-source Python repositories, sample projects from frameworks like LangChain, and some custom scripts we wrote for controlled testing. This mix gives us a variety of real-world and structured examples to work with.

### 2.2. Data Preprocessing
To make the data usable, we started by breaking down Python files into functions, classes, imports, and dependencies. We also split large files into smaller chunks using a text splitter so that our model can process them efficiently. The extracted relationships were then stored in a graph database, making it easier to run structured queries later. We plan to utilize vector embeddings so that we can quickly find similar pieces of code when answering user questions.

## 3. Graph Construction and RAG Implementation

### 3.1. How We’re Using RAG
Rather than just retrieving code snippets for answering questions, we’re using RAG to generate a structured graph that represents the relationships within the codebase. Right now, this graph is focused on literal dependencies, meaning it captures direct relationships like function calls, class inheritance, and imports. In the future, we plan to make the system more interactive, allowing users to ask specific questions and receive responses that combine both visualizations and explanations.

### 3.2. Two Approaches to Understanding Code
One of our main challenges is converting code into natural language descriptions in a way that helps developers understand not just the structure, but also the intent behind the code. We’re experimenting with two approaches:

1. **Literal Relationship Extraction**: This method focuses on mapping direct relationships within the code, such as which functions call each other or which classes inherit from others. This is useful for understanding the flow of execution.

2. **Conceptual Graph Representation**: Instead of just mapping direct dependencies, this approach aims to capture higher-level relationships between classes, files, and modules. The goal is to provide a more abstract understanding, like recognizing that “this group of classes handles authentication” rather than just showing individual function calls. This could eventually help the system generate better explanations for users.

In the long run, we want the system to use both methods, so it can provide both low-level structural details and high-level conceptual insights when analyzing a repository.

### 3.3. Agent-Driven Retrieval and Synthesis

Our system uses both GraphRAG and vector-based retrieval to provide more comprehensive code analysis. A Supervisor Agent coordinates these methods, determining when to query the graph database for structural relationships, the vector database for semantic context, or both in combination.

For example, when analyzing connections between Class A and Class B, the system first retrieves dependencies, function calls, and inheritance structures from GraphRAG. It then refines the search using vector embeddings to find relevant documentation or similar code snippets. This combined approach ensures responses capture both the explicit structure and broader context of the codebase, leading to clearer and more insightful explanations.

## 4. Progress So Far

So far, we’ve built the core system for parsing code and turning it into a graph. We’ve also integrated the LangChain RAG pipeline to retrieve relevant sections of code and connected everything to Memgraph so we can run structured queries. Initial tests on Python repositories have been successful, and we’ve also built some basic visualizations to display code relationships.

## 5. Next Steps
There are still a few key improvements we want to make. First, we need to refine how we extract and structure relationships within the codebase to make retrieval more effective. We also want to improve how the system generates natural language explanations, making them clearer and more useful. Another big goal is to integrate the conceptual graph, so that we can start capturing higher-level code structures. Eventually, we also want to add user interaction, where developers can ask the system questions and receieve either visualizations or explanations in response.

## 6. Conclusion
In this milestone, we set up the core components needed to analyze and retrieve code efficiently. Our system can now extract relationships from a repository, find relevant sections, and generate structured explanations. Moving forward, we’ll focus on making retrieval more accurate, improving explanation clarity, and designing a better user experience. With these improvements, we hope to create a powerful tool that makes understanding large codebases much easier, ultimately creating an augmented README.md file. 