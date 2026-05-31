# prepare data

import os
import fitz
import json

from milvus_client import milvus_client

def extract_pdf_text(data_dir, file_name, file_json):
    document = fitz.open(os.path.join(data_dir, file_name))
    pages = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        page_text = page.get_text()
        pages.append({
            "page_num": page_num,
            "text": page_text
        })

    with open(os.path.join(data_dir, file_json), "w", encoding="utf-8") as file:
        json.dump(pages, file, indent=4, ensure_ascii=False)

def generate_embeddings(data_dir, file_json, embeddings_json, model):
    pages = []
    with open(os.path.join(data_dir, file_json), "r") as file:
        data = json.load(file)

    for page in data:
        pages.append(page["text"])

    embeddings = model.encode(pages)

    embeddings_paginated = []
    for page_num in range(len(embeddings)):
        embeddings_paginated.append({"page_num": page_num, "embedding": embeddings[page_num].tolist()})

    with open(os.path.join(data_dir, embeddings_json), "w") as file:
        json.dump(embeddings_paginated, file, indent=4, ensure_ascii=False)


def insert_embeddings(data_dir, file_json, embeddings_json, client=milvus_client):
    rows = []
    with open(os.path.join(data_dir, file_json), "r") as t_f, open(os.path.join(data_dir, embeddings_json), "r") as e_f:
        text_data, embedding_data = json.load(t_f), json.load(e_f)
        text_data =  list(map(lambda d: d["text"], text_data))
        embedding_data = list(map(lambda d: d["embedding"], embedding_data))

        for page, (text, embedding) in enumerate(zip(text_data, embedding_data)):
            rows.append({"text":text, "embedding": embedding})

    client.insert(collection_name="rag_texts_and_embeddings", data=rows)


# insert_embeddings(file_json, embeddings_json)

# load inserted data into memory
# milvus_client.load_collection("rag_texts_and_embeddings")



# search
def search(model, query, client=milvus_client):
    embedded_query = model.encode(query).tolist()
    result = client.search(
        collection_name="rag_texts_and_embeddings",
        data=[embedded_query],
        limit=1,
        search_params={"metric_type": "L2"},
        output_fields=["text"]
    )
    return result


# result = search(model, query="Czym jest sztuczna inteligencja")
