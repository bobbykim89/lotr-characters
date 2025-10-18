import json
from tqdm import tqdm
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from composables.files import open_json_file
from composables.search import llm, format_hits_response

def format_rag_prompt (query: str, search_results: list[dict[str,str]]):
    raw_user_prompt = """
Context from database:
{retrieved_context}

User question:
{user_question}

Answer the question using ONLY the context above.
""".strip()
    
    system_prompt = """
You are a helpful lore expert on J.R.R. Tolkien's Middle-earth. 
You can only answer questions about characters using the provided context retrieved from the database. 
The context includes structured information such as: name, race, titles, realm, family relations, birth and death dates, and short descriptions.

Guidelines:
- If the answer is found in the context, respond clearly and directly.
- If the answer is not in the context, say you don’t know or that the information was not provided.
- Do not invent new facts outside the context.
- Keep your answers concise, but include all relevant details from the context.
- If the user asks for speculation (e.g., "what would happen if X met Y?"), you can summarize based only on what the context says about their traits.
""".strip()
    
    user_prompt = raw_user_prompt.format(retrieved_context=search_results, user_question=query).strip()
    return user_prompt, system_prompt

## Evaluation prompt created optimized for gpt-4o-mini
def format_eval_prompt (payload: dict[str,str])-> tuple[str, str]:
    raw_user_prompt = """
Evaluate the following RAG output.
{{
  "question": "{question}",
  "context": "{context}",
  "answer": "{answer}"
}}
""".strip()

    system_prompt = """
You are an impartial evaluator assessing the quality of a RAG (Retrieval-Augmented Generation) system that answers questions about J.R.R. Tolkien’s Middle-earth characters.

You will receive a JSON input with the following fields:
{
  "question": "<user query>",
  "context": "<retrieved context>",
  "answer": "<model-generated answer>"
}

Your task is to evaluate how well the answer satisfies the question, using only the information in the context.

Evaluate on four criteria:
1. Relevance — Does the answer directly address the question?
2. Groundedness — Are all facts supported by the provided context (no hallucinations)?
3. Completeness — Does the answer include all key details from the context?
4. Faithfulness — Does it follow the system rules (concise, factual, no invention, admits missing info)?

Scoring Guide (0–3 for each):
- 3: Excellent — fully meets the criterion
- 2: Fair — mostly correct, minor omissions or minor unsupported detail
- 1: Weak — noticeable errors, missing or irrelevant info
- 0: None — fails completely or contradicts context

Your output must be a single valid JSON object:
{
  "relevance": <0–3>,
  "groundedness": <0–3>,
  "completeness": <0–3>,
  "faithfulness": <0–3>,
  "comments": "<1–2 sentence summary of reasoning>"
}

Output only the JSON object — no markdown, no extra text.
""".strip()
    user_prompt = raw_user_prompt.format(question=payload.get('question'), context=payload.get('context'), answer=payload.get('answer')).strip()
    
    return user_prompt, system_prompt

def rag_eval_with_retrieval_results(data: dict):
    search_result = data.get('search_results')
    question = data.get('question')
    formatted_search_result = format_hits_response(hits=search_result)
    rag_user_prompt, rag_sys_prompt = format_rag_prompt(query=question, search_results=formatted_search_result)
    answer = llm(user_prompt=rag_user_prompt, system_prompt=rag_sys_prompt)
    payload = {
        "question": question,
        "context": search_result,
        "answer": answer
    }
    eval_user_prompt, eval_sys_prompt = format_eval_prompt(payload=payload)
    res = llm(user_prompt=eval_user_prompt, system_prompt=eval_sys_prompt)

    if type(res) == str:
        json_res = json.loads(res)
        return {"question": question, "answer": answer, **json_res}
    else:
        return {"question": question, "answer": answer, **res}
    
def generate_rag_eval_result_with_retrieval_results(data: list[dict]):
    eval_results = []
    for retrieval_result in tqdm(data, desc="Processing documents"):
        result = rag_eval_with_retrieval_results(data=retrieval_result)
        eval_results.append(result)
    return eval_results

def analyze_evaluation_result (file_path: str | Path):
    eval_data: list[dict] = open_json_file(file_path=file_path)
    num_entries = len(eval_data)
    
    relevance_total = 0
    groundedness_total = 0
    completeness_total = 0
    faithfulness_total = 0

    for entry in tqdm(eval_data, desc="Processing data"):
        relevance = entry.get("relevance")
        groundedness = entry.get("groundedness")
        completeness = entry.get("completeness")
        faithfulness = entry.get("faithfulness")

        relevance_total += relevance
        groundedness_total += groundedness
        completeness_total += completeness
        faithfulness_total += faithfulness
    
    avg_relevance = relevance_total / num_entries
    avg_groundedness = groundedness_total / num_entries
    avg_completeness = completeness_total / num_entries
    avg_faithfulness = faithfulness_total / num_entries
    total_avg_score = (relevance_total + groundedness_total + completeness_total + faithfulness_total) / (num_entries * 4)

    print("""
Evaluation using gpt-4o-mini
Evaluate on four criteria:
1. Relevance — Does the answer directly address the question?
2. Groundedness — Are all facts supported by the provided context (no hallucinations)?
3. Completeness — Does the answer include all key details from the context?
4. Faithfulness — Does it follow the system rules (concise, factual, no invention, admits missing info)?

Scoring Guide (0–3 for each):
- 3: Excellent — fully meets the criterion
- 2: Fair — mostly correct, minor omissions or minor unsupported detail
- 1: Weak — noticeable errors, missing or irrelevant info
- 0: None — fails completely or contradicts context
""")
    print(f"Number of entries: {num_entries}")
    print(f"Average Relevance Score: {avg_relevance}")
    print(f"Average Groundedness Score: {avg_groundedness}")
    print(f"Average Completeness Score: {avg_completeness}")
    print(f"Average Faithfulness Score: {avg_faithfulness}")
    print(f"Total Average Score: {total_avg_score}")