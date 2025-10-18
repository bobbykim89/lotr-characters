import pandas as pd
from rag_evaluation_fn import analyze_evaluation_result
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from composables.files import open_json_file

# print result of LLM Evaluation using LLM as a Judge method for gpt-4o-mini
evaluation_results_gpt_4o_mini_path = project_root / "src" / "assets" / "evaluation_results_gpt_4o_mini.json"
eval_results = open_json_file(file_path=evaluation_results_gpt_4o_mini_path)
analyze_evaluation_result(file_path=evaluation_results_gpt_4o_mini_path)
gpt_4o_mini_eval_df = pd.DataFrame(data=eval_results)
gpt_4o_mini_eval_df.to_csv(project_root / "src" / "assets" / "evaluation_results_gpt_4o_mini.csv")