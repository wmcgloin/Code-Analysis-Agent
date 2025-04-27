# Code Analysis Agent Demo Script

## Introduction (30 seconds)
"Hello everyone! Today I'm excited to demonstrate our Code Analysis Agent, a tool we've built to help developers quickly understand unfamiliar codebases. Our agent combines the power of LLMs with graph databases to transform code into queryable knowledge."

## Setup & Repository Loading (1 minute)
"Let me show you how it works. As you can see, the interface loads with a clean, simple design. On the sidebar, we have our repository input section. I'll paste in a GitHub URL for a Python project - let's use a medium-sized open-source library that many of you might be familiar with.

>[!NOTE]
> Alter repo depends on run time. 

After entering the URL and clicking 'Clone Repository', the system pulls down the code and displays the file structure. Now I can select which folders I want to analyze. For this demo, I'll focus on the main source code directory.

Once I've selected the folders, I'll click 'Initialize Database' to start the processing pipeline."

## Graph Construction & Macro Agent (1 minute)
"The system is now parsing the Python files, extracting classes, functions, and their relationships. It's using LLMs to generate natural language descriptions of each component and building a graph structure in Neo4j.

And now we have the query engine ready! This means the agents are initialized and ready to handle user input. Let's try a few prompts.

First let's see a high-level view of the codebase. This activates the LangGraph router, which invokes the macro agent to observe the codebase and generate a high-level diagram.

This diagram shows the high-level logical components and their relationships.
I can interact with this graph - zooming, panning, and clicking on nodes to see more details about each component. We've also provided a option to download the diagram as html for offline viewing."

## Querying the Codebase & Micro Agent (1.5 minutes)
"Now for the really powerful part - I can ask natural language questions about this codebase. For example let's say 'how does this function relate to that module?'

Now the system is processing my query through its LangGraph router. For this question, it's using the micro agent to provide a detailed relationship graph between the two entities.

Here's the response, which identifies the entities and explains their purpose. Now we can prompt the system to generate a diagram for it. On our sidebar a small diagram is displayed showing the semantic relationship between them, and how they correlates with others. I can of course also download the diagram."

## Advanced Features (30 seconds)
"The system supports follow-up questions too. If I ask 'Which of these has the most methods?', it remembers the context from my previous query and provides information about the class with the most methods."

>[!NOTE]
> Untested.

## Conclusion (30 seconds)
"And that's our Code Analysis Agent! In just a few minutes, we've gone from a raw GitHub repository to having detailed insights about the codebase structure, relationships, and functionality - all through natural language interaction.

This tool helps developers quickly understand new codebases, supports architectural decision-making, and makes code exploration more intuitive. Thank you for watching our demo!"
