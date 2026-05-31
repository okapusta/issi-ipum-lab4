import os
from google import genai

from milvus import search

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_KEY)

MODEL = "gemini-2.0-flash"

def generate_response(prompt: str):
    try:
        # Send request to Gemini 2.0 Flash API and get the response
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def build_prompt(context: str, query: str) -> str:
    return f"""
        You are a helpful assistant. Answer the question using only the context below.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """.strip()



def rag(model, query: str) -> str:
    results = search(model, query)

    context = "\n\n".join(
        r["entity"]["text"]
        for r in results[0]
    )[:155]

    prompt = build_prompt(context, query)
    print(prompt)
    return generate_response(prompt)




# from sentence_transformers import SentenceTransformer
# VECTOR_LENGTH = 768
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer

# model_name = "ipipan/silver-retriever-base-v1.1"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SentenceTransformer(model_name, device=device)
# from gemini import rag

# result = rag(model, "Czym jest sztuczna inteligencja")
