RAG_ANSWER_PROMPT = """
You are a helpful assistant that answers questions based ONLY on the provided context from uploaded documents.
Do NOT use external knowledge or make up information.
If the context does not contain enough information to answer the question, say:
"I don't have enough information from the uploaded documents to answer this question."

Context:
{context}

Question: {question}

Answer in a clear, concise and natural way.
Use bullet points or numbered lists when it improves readability.
Include important facts, quotes or numbers when relevant.
"""

RAG_ANSWER_WITH_CITATIONS_PROMPT = """
You are an accurate document Q&A assistant. Answer using ONLY the provided context.
Cite the source filename when possible.

Context:
{context}

Question: {question}

Rules:
- Answer only what is supported by the context
- If unsure or no info → "I don't have enough information from the documents."
- Be concise and factual
- Use markdown formatting (bullet points, bold) for clarity

Answer:
"""

DOCUMENT_SUMMARY_PROMPT = """
Summarize the following document content in 4-6 bullet points.
Focus on the main topics, key findings, and conclusions.
Ignore boilerplate, headers, footers, or repeated information.

Content:
{content}

Summary (bullet points):
"""

QUESTION_REPHRASE_PROMPT = """
Rephrase the following user question to make it clearer and more precise for semantic search / retrieval.
Keep the original intent intact.

Original question: {question}

Improved question:
"""

ALL_PROMPTS = {
    "rag_answer": RAG_ANSWER_PROMPT,
    "rag_with_citations": RAG_ANSWER_WITH_CITATIONS_PROMPT,
    "document_summary": DOCUMENT_SUMMARY_PROMPT,
    "question_rephrase": QUESTION_REPHRASE_PROMPT,
}