from pprint import pprint

import pandas as pd
from langchain.vectorstores import Chroma
import chromadb

from streamlit_app.llms.clients import OpenAILLMSClient

llm_client = OpenAILLMSClient()

from chromadb.config import Settings

# Configure the Chroma client with persistence
# Add a document to the collection
df = pd.read_json("streamlit_app/data/acris_real_property_legals_processed.json", orient="records",
                    lines=True)


# pop the embeddings so that it doesn't get added to metadata
embeddings = df.pop('text_embedding').values.tolist()

metadata = df.to_dict(orient='records')

ids = df['DOCUMENT ID'].values.tolist()

documents = df['text'].values.tolist()


class ChromaDB:
    def __init__(self, collection_name: str, persist_directory: str):
        """
        Initialize the ChromaDB vector store.
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.collection.upsert(
            ids=ids,  # Unique identifiers
            metadatas=metadata,  # Preprocessed metadata
            embeddings=embeddings,  # Embeddings
            documents=documents  # Original documents
        )
        print(f"Added {len(df)} records to the ChromaDB collection.")


    def search_document(self, query: str, k: int = 3):

        query_embedding = llm_client.get_embedding(query)
        """
        Search for documents similar to the query.
        Args:
            query (str): Query text.
            k (int): Number of results to retrieve.
        Returns:
            list: Search results with scores.
        """
        search_results = self.collection.query(
            query_embedding,
            n_results=k
        )
        pprint(search_results)

        metadatas = search_results['metadatas'][0]

        content = "\n\n\n".join([result['text'] for result in metadatas])

        search_as_a_function_call = [{
            "role": "function",
            "name": "search_document",
            "content": content,
        }]

        return search_as_a_function_call

