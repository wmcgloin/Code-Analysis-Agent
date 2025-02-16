# End-of-Semester Project Deliverables

## Overview
The end-of-semester project is an opportunity for students to demonstrate their understanding of generative AI by building a fully functional application. The project should follow a structured approach, similar to an academic research paper, while also showcasing implementation details, findings, and a working demo. 

Each project must include a report (see [`submissions requirements`](#submission-requirements)) structured with the following sections:

---

## 1. Title and Abstract
- A concise title that reflects the project’s focus.
- A 250–300 word abstract summarizing the problem statement, approach, key techniques used, and main findings.
- This submission constitutes Milestone 1 and must be approved by the professor. Feedback must be incorporated before proceeding.

## 2. Introduction
- Overview of the problem being addressed.
- Motivation for choosing this problem and its relevance in the field of Generative AI.
- Research questions or hypotheses (if applicable).

## 3. Data Source and Preparation
- Description of the data sources used (public datasets, proprietary data, scraped data, etc.).
- Data preprocessing steps, including cleaning, augmentation, and transformations.
- Justification for data selection and preprocessing choices.
- This is part of Milestone 2.

## 4. Retrieval-Augmented Generation (RAG) (Mandatory, unless approved by professor)
- Explanation of how RAG was implemented.
- Details of the embedding models used (e.g., OpenAI, Cohere, BGE, custom fine-tuned models).
- Discussion on retrieval techniques (e.g., vector search, hybrid search, graph-based retrieval).
- Large Language Models (LLMs) used for answer generation and their reasoning.
- How query-to-context matching was optimized.
- This is part of Milestone 2.

## 5. Agents (Mandatory, unless approved by professor)
- Architecture of AI agents used.
- Frameworks used (e.g., LangChain, LlamaIndex, Haystack).
- Types of agents (e.g., tool-using agents, autonomous agents, planning agents).
- How agents interact with external tools and APIs.
- This is part of Milestone 3.

## 6. Models and Technologies Used
- Discussion on the models utilized (LLMs, embedding models, classifiers, etc.).
- Comparison of different models and justification for final choices.
- Any inference optimizations applied (e.g., quantization, distillation, multi-adapter swapping).

## 7. Fine-Tuning (if applicable)
- Whether fine-tuning was performed.
- If yes, details of dataset preparation, model training, evaluation, and computational requirements.
- If no, justification for choosing not to fine-tune.

## 8. Tools and Frameworks
- Programming languages and frameworks used (e.g., Python, PyTorch, TensorFlow, Hugging Face, OpenAI API).
- Cloud platforms and any relevant services (e.g., Amazon Bedrock, Amazon SageMaker, Vector DBs like FAISS), open-source (e.g. LangChain).
- Version control, CI/CD, or deployment considerations.

## 9. Evaluation of Effectiveness
- Metrics used for evaluation (e.g., BLEU, ROUGE, MRR, Precision/Recall for retrieval, perplexity for LLMs).
- Benchmarking comparisons against baselines.
- User studies, feedback, or qualitative assessments.
- This is part of Milestone 3.

## 10. Responsible AI Considerations
- Bias and fairness checks conducted.
- Any measures taken to reduce hallucinations and ensure factual consistency.
- Privacy considerations (e.g., PII removal, data anonymization).
- Model safety measures and ethical implications.

## 11. Findings and Insights
- Key takeaways from the project.
- Unexpected challenges and how they were addressed.
- Performance trade-offs and scalability considerations.

## 12. Demo
- A 5-minute working demonstration of the project (video link or live demo link).
- Explanation of key features in the demo.
- User interface and interaction details.

## 13. Conclusion and Future Work
- Summary of the project and its impact.
- Limitations of the current implementation.
- Potential future improvements and extensions.

## 14. References
- List of academic papers, blog posts, documentation, and other resources referenced.

## 15. Appendix (if needed)
- Additional code snippets, extended data analysis, or supplementary material.

---

## Milestones
1. **Milestone 1**: Title and Abstract Submission (Must be approved by professor, feedback must be incorporated before proceeding). **Due date: Friday, 28th February, 2025**.
2. **Milestone 2**: Data Preparation and RAG Implementation. **Due date: Tuesday, 18th March, 2025**.
3. **Milestone 3**: Agents and Evaluation. **Due date: Tuesday, 11th April, 2025**.
4. **Final Milestone**: Project Submission. **Due date: Friday, 25th April, 2025**.
4. **Final Presentation**: Project Presentation. **Due date: Tuesday, 29th April, 2025**.

After projects have been submitted, there will be an ungraded project presentation. This will typically require a PowerPoint presentation and a live demo. Each group will have **15 minutes** to present their work.

## Submission Requirements
- **Format:** Markdown/PDF report + GitHub repository (this repository) with code. Use [`Quarto`](https://quarto.org/)  or [`MkDocs`](https://www.mkdocs.org/) to create your report and host it on GitHub pages for this repo. It is up to you if you want to host this is a single page website or in a different format, whatever best helps communicate the deliverables listed above. **This is a graduate class therefore attention to detail is a must, for example cross references in report to sections of code, tables, visualizations and screenshots to communicate key results, coherent and concise writing free of grammatical errors are all baseline level requirements**.
- **Due Date:** Tuesday, April 29th, 2025
- **Evaluation Criteria:**
  - Technical depth and correctness (40%)
  - Novelty and innovation (30%)
  - Clarity of explanation and documentation (20%)
  - Quality of the working demo (10%)



