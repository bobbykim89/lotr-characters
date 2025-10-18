import json
import sys
from pathlib import Path
from tqdm import tqdm
from os import environ
from anthropic import Anthropic
from rag_evaluation_fn import format_rag_prompt

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from composables.files import open_json_file, save_json_file
from composables.search import llm, format_hits_response

# Modified fns that are specific for RAG using Anthropic

ANTHROPIC_API_KEY = environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

## Evaluation prompt created optimized for claude-3-5-haiku-20241022
def format_eval_prompt (payload: dict[str,str])-> tuple[str, str]:
    raw_user_prompt = """
Evaluate the following RAG output:

Question: {question}

Context: {context}

Answer: {answer}
""".strip()

    system_prompt = """
You are an impartial evaluator assessing RAG (Retrieval-Augmented Generation) system outputs for questions about J.R.R. Tolkien's Middle-earth characters.

Evaluate each answer on four criteria using a 0-3 scale:

1. Relevance (0-3): Does the answer directly address the question?
   - 3: Fully addresses the question
   - 2: Mostly relevant with minor tangents
   - 1: Partially relevant, significant gaps
   - 0: Irrelevant or off-topic

2. Groundedness (0-3): Are all facts supported by the context?
   - 3: All claims supported, no hallucinations
   - 2: Mostly grounded, one minor unsupported detail
   - 1: Multiple unsupported claims
   - 0: Significant hallucinations or contradicts context

3. Completeness (0-3): Does the answer include key details from context?
   - 3: All important information included
   - 2: Most key details present, minor omissions
   - 1: Missing significant information
   - 0: Incomplete or vague

4. Faithfulness (0-3): Is the answer concise, factual, and honest about limitations?
   - 3: Concise, factual, admits gaps appropriately
   - 2: Mostly faithful, slightly verbose or assumes minor details
   - 1: Invents information or doesn't admit uncertainty
   - 0: Violates multiple guidelines

Return only a JSON object with this exact structure:
{
  "relevance": <integer 0-3>,
  "groundedness": <integer 0-3>,
  "completeness": <integer 0-3>,
  "faithfulness": <integer 0-3>,
  "comments": "<brief reasoning in 1-2 sentences>"
}

No markdown formatting, no additional text, just the JSON object.
""".strip()
    user_prompt = raw_user_prompt.format(question=payload.get('question'), context=payload.get('context'), answer=payload.get('answer')).strip()
    
    return user_prompt, system_prompt

def llm_anthropic(user_prompt: str, system_prompt: str):
    message = anthropic_client.messages.create(
        model="claude-3-5-haiku-20241022",
        system=system_prompt,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    return message.content[0].text

def rag_eval_with_retrieval_results_anthropic(data: dict):
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
    res = llm_anthropic(user_prompt=eval_user_prompt, system_prompt=eval_sys_prompt)

    if type(res) == str:
        json_res = json.loads(res)
        return {"question": question, "answer": answer, **json_res}
    else:
        return {"question": question, "answer": answer, **res}
    
def generate_rag_eval_result_with_retrieval_results_anthropic(data: list[dict]):
    eval_results = []
    for retrieval_result in tqdm(data, desc="Processing documents"):
        result = rag_eval_with_retrieval_results_anthropic(data=retrieval_result)
        eval_results.append(result)
    return eval_results

def analyze_evaluation_result_anthropic (file_path: str | Path):
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
Evaluation using claude-3-5-haiku-20241022
Evaluate on four criteria:
1. Relevance (0-3): Does the answer directly address the question?
   - 3: Fully addresses the question
   - 2: Mostly relevant with minor tangents
   - 1: Partially relevant, significant gaps
   - 0: Irrelevant or off-topic

2. Groundedness (0-3): Are all facts supported by the context?
   - 3: All claims supported, no hallucinations
   - 2: Mostly grounded, one minor unsupported detail
   - 1: Multiple unsupported claims
   - 0: Significant hallucinations or contradicts context

3. Completeness (0-3): Does the answer include key details from context?
   - 3: All important information included
   - 2: Most key details present, minor omissions
   - 1: Missing significant information
   - 0: Incomplete or vague

4. Faithfulness (0-3): Is the answer concise, factual, and honest about limitations?
   - 3: Concise, factual, admits gaps appropriately
   - 2: Mostly faithful, slightly verbose or assumes minor details
   - 1: Invents information or doesn't admit uncertainty
   - 0: Violates multiple guidelines
""")
    print(f"Number of entries: {num_entries}")
    print(f"Average Relevance Score: {avg_relevance}")
    print(f"Average Groundedness Score: {avg_groundedness}")
    print(f"Average Completeness Score: {avg_completeness}")
    print(f"Average Faithfulness Score: {avg_faithfulness}")
    print(f"Total Average Score: {total_avg_score}")

# Running evaluations using claude-3-5-haiku-20241022 using LLM as a judge method
retrieval_search_results_path = project_root / "src" / "assets" / "retrieval_search_results.json"
raw_search_results = open_json_file(file_path=retrieval_search_results_path)
eval_results_anthropic = generate_rag_eval_result_with_retrieval_results_anthropic(data=raw_search_results)
evaluation_results_claude_3_5_haiku_path = project_root / "src" / "assets" / "evaluation_results_claude_3_5_haiku.json"
save_json_file(data=eval_results_anthropic, file_path=evaluation_results_claude_3_5_haiku_path)
analyze_evaluation_result_anthropic(file_path=evaluation_results_claude_3_5_haiku_path)