# GenAI project ideas


## Code Visualizer/Debugger for devops

A tool that can visualize the codebase of a project.

### Features:

- draw function call graph
- draw logic flow graph
- draw import graph
- propose graph for dev
- highlight graph from user input
- suggest code structurefrom graph
- track variable usage

### Existing tools:

- [PyCallGraph]://github.com/Lewiscowles1986/py-call-graph?tab=readme-ov-file)
- [graphviz]://github.com/CodeFreezr/awesome-graphviz)


### Tech stack

#### LLM
- langchain
   - LLMGraphTransformer from experimental package. ex: [structured_output.py](../src/structured_output.py) (line 46)
   - with_structured_output

#### Visualizer
- visjs [index.html](../src/index.html)


### Challenges

- langchain
   - LLMGraphTransformer does not perform well when analyzing code base. 
   - 
