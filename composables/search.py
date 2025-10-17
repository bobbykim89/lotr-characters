from openai import OpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import json
import requests
from os import environ

load_dotenv()

QDRANT_URL = environ.get('QDRANT_URL')
QDRANT_API_KEY = environ.get('QDRANT_API_KEY')
COLLECTION_NAME = 'lotr-characters'
EMBEDDING_DIMENSION = 512
JINA_EMBEDDING_MODEL = "jina-embeddings-v4"
JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = environ.get('JINA_API_KEY')
QUERYING_TASK = "retrieval.query"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.5

openai_client = OpenAI()
qd_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def create_jina_embedding(input_text: str)-> list:
    """
    Create embedding using Jina API
    Returns a single embedding vector (list of floats)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "input": [input_text],
        "model": JINA_EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSION,
        "task": QUERYING_TASK,
        "late_chunking": True,
    }
    try:
        res = requests.post(url=JINA_URL, headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            embedding = res.json()["data"][0]["embedding"]
            return embedding
        else:
            raise Exception(f"Jina API error: {res.status_code} - {res.text}")
    except requests.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")
    

def search(query: str, limit: int = 5):
    """
    Updated search function to use Jina API for query embedding
    """
    try:
        # Create embedding for the search query using Jina API
        query_embedding = create_jina_embedding(input_text=query)
        
        query_points = qd_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        results = [{"id": point.id, "score": point.score, **point.payload} for point in query_points.points]
        return results
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return None
    
def format_hits_response(hits: list[dict[str, str|None]]):
    """Format the results into text to plug into chatGPT"""
    character_data = []
    for hit in hits:
        basic_fields = ['id', 'score', 'name', 'race', 'gender', 'realm', 'culture', 'birth', 'death', 'spouse', 'hair', 'height', 'biography', 'history']
        character = {}
        character.update([(field, hit[field]) for field in basic_fields if hit.get(field)])
        character_data.append(character)
    
    return json.dumps(character_data, indent=2)

def llm(user_prompt: str, system_prompt: str):
    """ llm function to call openAI with our specific prompts"""
    res = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=OPENAI_TEMPERATURE
    )
    return res.choices[0].message.content