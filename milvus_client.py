from pymilvus import MilvusClient

host = "localhost"
port = "19530"

milvus_client = MilvusClient(
    host=host,
    port=port
)
