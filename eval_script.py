import sys
import types
import os
import time

# --- 1. WORKAROUND FOR RAGAS / LANGCHAIN IMPORT ISSUES ---
dummy_module = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_module.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_module

import pandas as pd
from datasets import Dataset
from openai import OpenAI

# RAGAS Core & LLM / Embeddings Wrappers
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper

# RAGAS metrics including AnswerRelevancy
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall

# EduFocus pipeline imports
from utils.vector_store import load_vector_store, retrieve_similar_chunks
from utils.chat_engine import get_groq_chat_model
from utils.prompts import get_system_prompt, format_context_with_citations

# Use updated LangChain HuggingFace embeddings
from langchain_huggingface import HuggingFaceEmbeddings


# --- 2. DEFINE TEST DATA ---
test_questions = [
    "What are the four necessary conditions for a deadlock to occur in an operating system?",
    "What is the difference between Single Instance and Multi Instance Resource Allocation Graphs?",
    "What is the Ostrich Algorithm or Deadlock Ignorance strategy?",
    "What data structures are used to implement Banker's Algorithm?"
]

ground_truths = [
    "The four necessary conditions for deadlock are Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.",
    "In a Single Instance RAG, each resource has only one copy and a cycle strictly guarantees a deadlock. In a Multi Instance RAG, resources have multiple copies and a cycle is necessary but not a sufficient condition for deadlock.",
    "Deadlock Ignorance (Ostrich Algorithm) is an approach where the OS assumes deadlocks are so rare that it is not worth the performance cost to prevent or detect them, relying on a reboot if a deadlock happens.",
    "Banker's Algorithm uses four data structures: Available (1D array), Max (2D array), Allocation (2D array), and Need (2D array)."
]

questions = []
answers = []
contexts = []
latencies = []


# --- 3. INITIALIZE PIPELINE & GENERATE RESPONSES ---
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = load_vector_store(hf_embeddings)

if vector_store is None:
    raise ValueError("Vector DB not found! Make sure you indexed Deadlock.pdf into FAISS first.")

chat_model = get_groq_chat_model()
system_prompt = get_system_prompt("Beginner")

for q in test_questions:
    start_time = time.time()
    
    # Retrieve chunks from FAISS
    chunks = retrieve_similar_chunks(vector_store, q, k=3)
    chunk_texts = [c.page_content for c in chunks]
    
    # Format prompt specifically for Deadlock.pdf
    context_str = format_context_with_citations(chunks, "Deadlock.pdf")
    full_prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {q}"
    
    # Query model
    response = chat_model.invoke(full_prompt)
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    questions.append(q)
    answers.append(response.content)
    contexts.append(chunk_texts)
    latencies.append(latency_ms)


# --- 4. PREPARE DATASET FOR RAGAS ---
data_dict = {
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
}
dataset = Dataset.from_dict(data_dict)


# --- 5. SETUP EVALUATOR ENGINE WITH GROQ + HUGGINGFACE ---
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing!")

# 1) Setup Groq as the LLM judge via OpenAI client wrapper
groq_client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)
evaluator_llm = llm_factory(model="llama-3.3-70b-versatile", client=groq_client)

# 2) Wrap local HuggingFace embeddings for RAGAS metrics that require vector embeddings
ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Instantiate all 3 metrics with explicit evaluator LLM and Embeddings
metrics_list = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=ragas_embeddings),
    LLMContextRecall(llm=evaluator_llm)
]


# --- 6. EXECUTE EVALUATION ---
results = evaluate(
    dataset=dataset,
    metrics=metrics_list,
    llm=evaluator_llm,
    embeddings=ragas_embeddings
)

avg_latency = sum(latencies) / len(latencies)


# --- 7. PRINT RESULTS ---
print("\n--- OS DEADLOCK RAG EVALUATION RESULTS ---")
print(results)
print(f"Average Response Latency: {avg_latency:.0f} ms")