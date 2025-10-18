import json
from tqdm import tqdm
import time
import itertools
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from composables.files import open_json_file, save_json_file
from composables.search import search, llm

# open json file that contains data stored in qdrant
# duplicate of data stored in qdrant cloud, and formatted and saved in json for convenience.
qdrant_records_file_path = project_root / "src" / "assets" / "qdrant_records.json"
qdrant_records = open_json_file(file_path=qdrant_records_file_path)

def format_prompt (payload: dict[str,str])-> tuple[str, str]:
    raw_user_prompt = """
Payload:
{payload}
""".strip()

    system_prompt = """
You are an assistant that generates evaluation questions to test retrieval quality on character data.
You will receive a JSON object describing a fictional character, which may include fields such as name, race, gender, birth, death, spouse, realm, biography, and others.

Your task:
1. Read and understand the JSON payload carefully.
2. Generate exactly 5 diverse and specific questions that can be answered using the information in the payload.
3. Focus on factual, grounded details — such as relationships, timeline, characteristics, or key events mentioned in the biography.
4. Avoid trivial or repetitive questions.
5. Do not include any reasoning, explanations, or text outside the JSON array.

Output valid JSON only (no code blocks, no extra text):
["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]

If a field is null or missing, do not ask about it. If there is limited information, create general but relevant questions based on available content.
""".strip()
    user_prompt = raw_user_prompt.format(payload=payload).strip()
    
    return user_prompt, system_prompt

def format_records (data: list[dict])->list[dict]:
    """
    format character information
    """
    formatted_records = []
    for record in tqdm(data):
        basic_fields = ['name', 'race', 'gender', 'realm', 'culture', 'birth', 'death', 'spouse', 'hair', 'height', 'biography', 'history']
        character = {
            "id": record["id"]
        }
        character.update([(field, record["payload"][field]) for field in basic_fields if record["payload"].get(field)])
        formatted_records.append(character)

    return formatted_records

# generate questions using open ai based on system_prompt and user_prompt provided
def generate_question(ctx: dict[str,str])->dict:
    user_prompt, system_prompt = format_prompt(ctx)
    questions = llm(user_prompt=user_prompt, system_prompt=system_prompt)
    return {
        "id": ctx['id'],
        "questions": json.loads(questions)
    }


def generate_questions_and_save_json():
    records = format_records(data=qdrant_records)
    formatted_questions = []
    for record in tqdm(records):
        questions = generate_question(ctx=record)
        formatted_questions.append(questions)
    save_path = project_root / "src" / "assets" / "golden_questions.json"
    save_json_file(file_path=save_path, data=formatted_questions)

def get_formatted_search_result(golden_questions: list[dict]=None, previous_results=None, start_index: int=0, requests_per_minute: int=400):
    search_results = previous_results if previous_results is not None else []
    current_index = start_index

    # Calculate delay between requests to stay under rate limit
    delay_seconds = 60.0 / requests_per_minute

    try:
        for obj in tqdm(golden_questions, desc="Processing documents"):
            doc_id = obj["id"]
            for q_idx, question in enumerate(obj["questions"]):
                if current_index < start_index:
                    current_index += 1
                    continue

                try:
                    results = search(query=question, limit=5, threshold=0.3)
                    if results is None:
                        raise ValueError("Search returned None")
                    search_result = {
                        "id": doc_id,
                        "question": question,
                        "question_idx": q_idx,
                        "search_results": results
                    }
                    search_results.append(search_result)
                    current_index += 1

                    # Add delay to respect rate limit
                    time.sleep(delay_seconds)
                except Exception as e:
                    print(f"\n❌ Error at index {current_index}")
                    print(f"   Document ID: {doc_id}")
                    print(f"   Question {q_idx + 1}/{len(obj['questions'])}: {question}")
                    print(f"   Error: {type(e).__name__}: {str(e)}")
                    print(f"\n💾 Processed {len(search_results)} questions before failure")
                    print(f"   Returning (relevance_total, {current_index}) for resume")
                    return search_results, current_index
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user at index {current_index}")
        print(f"💾 Processed {len(search_results)} questions")
        return search_results, current_index
    
    return search_results, current_index

def filter_results(data: list[dict], filters: dict):
    """
    Filter search results based on limit and threshold.
    Args:
        data: List of result objects
        filters: Dictionary with 'limit' (int) and 'threshold' (float)
    Returns:
        Filtered data with same structure
    """
    limit = filters.get('limit')
    threshold = filters.get('threshold')

    filtered_data = []

    for entry in tqdm(data, desc=f"Filtering results satifying {filters}"):
        filtered_entry = entry.copy()
        results = entry['search_results']

        # apply limit (top x items)
        if limit is not None:
            results = results[:limit]
        # apply threshold filter
        if threshold is not None:
            results = [r for r in results if r.get('score', 0) > threshold]
        # update the entry with filtered results
        filtered_entry['search_results'] = results
        filtered_data.append(filtered_entry)
    
    return filtered_data

def get_strategies_list()-> list[dict]:
    ## list of strategies
    limits = [3, 4, 5]
    thresholds = [0.3, 0.5, 0.7]
    strategies = []

    for limit, threshold in itertools.product(limits, thresholds):
        strategy = {"limit": limit, "threshold": threshold}
        strategies.append(strategy)
    return strategies

def make_relevance_matrix(data: list[dict]):
    """
    Create relevance matrix based on given data
    Args:
        data: List of result objects
    Returns:
        relevance matrix nested list of boolean
    """
    relevance_total = []
    for obj in tqdm(data):
        obj_id = obj["id"]
        relevance = [result['id'] == obj_id for result in obj["search_results"]]
        relevance_total.append(relevance)
    
    return relevance_total

def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + (1 / (rank + 1))
    return total_score / len(relevance_total)

def evaluate(relevance_total):
    return {
        "hit_rate": hit_rate(relevance_total=relevance_total),
        "mrr": mrr(relevance_total=relevance_total)
    }

def generate_evaluations_per_strategy(search_results: list[dict], strategies: list[dict]):
    results = []
    for strategy in tqdm(strategies):
        filtered_results = filter_results(data=search_results, filters=strategy)
        relevance_total = make_relevance_matrix(data=filtered_results)
        eval = evaluate(relevance_total=relevance_total)
        formatted_evaluation = {**strategy, **eval}
        results.append(formatted_evaluation)
    return results

def print_evaluation_result(results: list[dict]):
    highest_hit_rate = max(results, key=lambda x: x['hit_rate'])
    highest_mrr = max(results, key=lambda x: x['mrr'])

    print(f"Highest Hit Rate: {highest_hit_rate}")
    print(f"Highest MRR: {highest_mrr}")