from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import uuid
import json
import requests
from os import environ
import tiktoken
import re
import time

load_dotenv()

QDRANT_URL = environ.get('QDRANT_URL')
QDRANT_API_KEY = environ.get('QDRANT_API_KEY')
COLLECTION_NAME = 'lotr-characters'
EMBEDDING_DIMENSION = 512
JINA_EMBEDDING_MODEL = "jina-embeddings-v4"
JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = environ.get('JINA_API_KEY')
INDEXING_TASK = "retrieval.passage"
QUERYING_TASK = "retrieval.query"
MAX_TOKENS = 8000

# init qdrant
qd_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

tokenizer = tiktoken.get_encoding("cl100k_base")

# read json file
with open('./assets/lotr_characters.json', 'r') as file:
    characters = json.load(file)

print(f"Loaded {len(characters)} entries.")

def count_token(text: str)-> int:
    return len(tokenizer.encode(text=text))

def create_jina_embedding(input_text: str, task = INDEXING_TASK)-> list:
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
        "task": task,
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
    
def truncate_text_smart(text: str, max_tokens: int = 8000)-> str:
    """
    Truncate text at full sentence boundaries without exceeding max_tokens.
    Keeps as many full sentences as possible.
    """
    if count_token(text=text) <= max_tokens:
        return text
    
    # Split into sentences (handles ., !, ? with possible whitespace)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    truncated_sentences = []
    total_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_token(sentence)
        if total_tokens + sentence_tokens <= max_tokens:
            truncated_sentences.append(sentence)
            total_tokens += sentence_tokens
        else:
            break
    
    return " ".join(truncated_sentences).strip()


def create_character_text_safe(character: dict, max_tokens: int = 7000)-> str:
    """
    Create character text formatted for embedding,
    truncated safely to fit within max_tokens.
    """
    text_parts = []
    
    # Always include name
    if character.get('name'):
        text_parts.append(f"Name: {character['name']}")

    # add other basic info
    basic_fields = ['race', 'gender', 'realm', 'culture', 'birth', 'death', 'spouse', 'hair', 'height']
    text_parts.extend(
        f"{field.title()}: {character[field]}"
        for field in basic_fields
        if character.get(field)
    )

    # Track token budget so far
    base_text = "\n".join(text_parts)
    base_tokens = count_token(base_text)
    remaining_tokens = max_tokens - base_tokens

    # try to include biography/history if space allows
    for field in ["biography", "history"]:
        content = character.get(field)
        if content and remaining_tokens > 0:
            truncated = truncate_text_smart(content.strip(), remaining_tokens)
            field_text = f"{field.title()}: {truncated}"
            tokens_used = count_token(field_text)

            if tokens_used <= remaining_tokens:
                text_parts.append(field_text)
                remaining_tokens -= tokens_used
    
    final_text = "\n".join(text_parts)

    # final hard cap (in case token calc was optimistic)
    if count_token(final_text) > max_tokens:
        final_text = truncate_text_smart(final_text, max_tokens=max_tokens)

    return final_text


def create_character_summary(character: dict, max_tokens: int = 500)-> str:
    """
    Create a concise summary for characters with very long descriptions
    """
    name = character.get('name', 'Unknown')
    summary_parts = [name]

    # add key identifiers
    if character.get('race'):
        summary_parts.append(f"a {character['race']}")
    if character.get('realm'):
        summary_parts.append(f"from {character['realm']}")    
    if character.get('culture'):
        summary_parts.append(f"of {character['culture']} culture")
    
    # extract first few sentences from biography/history
    content: str = character.get('biography') or character.get('history')
    if content is not None:
        tokens = tokenizer.encode(content)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        first_sentences = tokenizer.decode(tokens=tokens) + "..."
        summary_parts.append(first_sentences)
    
    return " - ".join(summary_parts)


def create_jina_embedding_batch_safe(input_texts: list, max_token_per_text: int = 6000) -> list:
    """
    Create embeddings for multiple texts with length safety checks
    """
    # First, ensure all texts are within safe limits
    safe_texts = []
    for text in input_texts:
        safe_text = truncate_text_smart(text=text, max_tokens=max_token_per_text)
        safe_texts.append(safe_text)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }
    data = {
        "input": safe_texts,
        "model": JINA_EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSION,
        "task": INDEXING_TASK,
        "late_chunking": True,
    }

    try:
        res = requests.post(url=JINA_URL, headers=headers, json=data, timeout=120)
        if res.status_code == 200:
            embeddings = [d["embedding"] for d in res.json()["data"]]
            return embeddings
        else:
            raise Exception(f"Jina API error: {res.status_code} - {res.text}")
    except requests.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")


def reinitiate_collection():
    is_collection_exist = qd_client.collection_exists(collection_name=COLLECTION_NAME)
    if is_collection_exist:
        qd_client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    print(f"Collection {COLLECTION_NAME} didn't exist, creating new one")
    qd_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION, # Dimensionality of the vectors
            distance=models.Distance.COSINE # Distance metric for similarity search
        )
    )
    print("Created the new collection")


"""
strategy: adaptive batching:
- Add texts to the current batch until you're near the token budget.
- If adding the next text would exceed the budget, start a new batch.
- For very large texts that nearly hit the limit on their own → process them individually.
"""

def upsert_to_qdrant_adaptive(max_tokens_per_batch: int = 7000, max_tokens_per_text: int = 6000):
    """
    Upsert to Qdrant with dynamic batch sizing based on token usage.
    Each batch is sized to stay under max_tokens_per_batch.
    Individual texts that nearly hit the limit are embedded one by one.
    """

    # Calculate delay between requests to stay under rate limit
    requests_per_minute=400
    delay_seconds = 60.0 / requests_per_minute

    if not qd_client.collection_exists(collection_name=COLLECTION_NAME):
        print(f'Collection {COLLECTION_NAME} does not exist.')
        return
    # prepare safe character texts
    print("Preparing safe character texts...")
    prepared_data = []
    for character in characters:
        try:
            text = create_character_text_safe(character=character, max_tokens=max_tokens_per_text)
            token_count = count_token(text)
            prepared_data.append({
                "character": character,
                "text": text,
                "token_count": token_count
            })
        except Exception as e:
            print(f"Error preparing text for {character.get('name', 'Unknown')}: {str(e)}")
    
    if not prepared_data:
        print("No valid character data to process")
        return
    
    print(f"Prepared {len(prepared_data)} characters for processing")

    # Sort shortest first → more likely to fill batches efficiently
    prepared_data.sort(key=lambda x: x['token_count'])

    # build dynamic batches
    batches = []
    current_batch = []
    current_tokens = 0

    for entry in prepared_data:
        tc = entry['token_count']

        # if one text alone is too large, process separately as its own batch
        if tc > max_tokens_per_batch:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([entry])
            continue

        # if adding this entry would exceed batch limit, start a new batch
        if current_tokens + tc > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(entry)
        current_tokens += tc

    if current_batch:
        batches.append(current_batch)
    
    print(f"built {len(batches)} batches for processing")
    
    # process batches and collect points
    all_points = []
    total_processed = 0

    for batch_num, batch in enumerate(batches, start=1):
        texts = [e['text'] for e in batch]

        print(f"processing batch {batch_num}/{len(batches)} with {len(batch)} entries, total tokens ≈ {sum(e['token_count'] for e in batch)}")

        try:
            embeddings = create_jina_embedding_batch_safe(texts, max_token_per_text=max_tokens_per_text)
            time.sleep(delay_seconds)
        except Exception as batch_error:
            print(f"batch {batch_num} failed: {str(batch_error)}")
            # fallback: process individually
            embeddings = []
            
            for entry in batch:
                try:
                    emb = create_jina_embedding(entry['text'])
                    embeddings.append(emb)
                except Exception as e:
                    print(f"failed individual embedding for {entry['character'].get('name', 'Unknown')}: {str(e)}")
                    embeddings.append(None)
                
        # convert to qdrant points
        for entry, embedding in zip(batch, embeddings):
            if embedding is not None:
                point = models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=embedding,
                    payload={
                        **entry["character"],
                        "embedded_text": entry["text"],
                        "token_count": entry["token_count"]
                    }
                )
                all_points.append(point)
                total_processed += 1
    # final upsert in one shot
    if all_points:
        try:
            qd_client.upsert(collection_name=COLLECTION_NAME, points=all_points)
            print(f"successfully upserted {total_processed}/{len(prepared_data)} entries to qdrant")
        except Exception as e:
            print(f"final upsert failed: {str(e)}")
    else:
        print("no valid embeddings to upsert")


def search(query: str, limit: int = 1):
    """
    Updated search function to use Jina API for query embedding
    """
    try:
        # Create embedding for the search query using Jina API
        query_embedding = create_jina_embedding(input_text=query, task=QUERYING_TASK)
        
        query_points = qd_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )
        results = [point.payload for point in query_points.points]

        return results
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return None
    

def search_with_score_threshold(query: str, limit: int = 5, score_threshold: float = 0.7):
    """
    Enhanced search function with similarity score filtering
    """
    try:
        query_embedding = create_jina_embedding(input_text=query, task=QUERYING_TASK)

        query_points = qd_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold
        )
        results = [point.payload for point in query_points.points]
        return results
    except Exception as e:
        print(f"Error during search with threshold: {str(e)}")
        return None
    

reinitiate_collection()
upsert_to_qdrant_adaptive()