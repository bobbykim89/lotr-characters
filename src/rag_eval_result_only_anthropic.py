import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from composables.files import open_json_file

# copy of fn in rag_eval_anthropic.py
def analyze_evaluation_result_anthropic (file_path: str):
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

# print result of LLM Evaluation using LLM as a Judge method for claude-3-5-haiku-20241022
evaluation_results_gpt_4o_mini_path = project_root / "src" / "assets" / "evaluation_results_claude_3_5_haiku.json"
eval_results = open_json_file(file_path=evaluation_results_gpt_4o_mini_path)
analyze_evaluation_result_anthropic(file_path=evaluation_results_gpt_4o_mini_path)
anthropic_eval_df = pd.DataFrame(data=eval_results)
anthropic_eval_df.to_csv(project_root / "src" / "assets" / "evaluation_results_claude_3_5_haiku.csv")