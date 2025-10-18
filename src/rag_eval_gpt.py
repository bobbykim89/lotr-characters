import sys
from pathlib import Path
import pandas as pd
from rag_evaluation_fn import generate_rag_eval_result_with_retrieval_results, analyze_evaluation_result

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from composables.files import open_json_file, save_json_file
from composables.search import llm, format_hits_response
from composables.data_processing import format_list_in_batch

# run AI evaluation using LLM as Judge method
golden_questions_path = project_root / "src" / "assets" / "golden_questions.json"
golden_questions = open_json_file(file_path=golden_questions_path)
retrieval_search_results_path = project_root / "src" / "assets" / "retrieval_search_results.json"
raw_search_results = open_json_file(file_path=retrieval_search_results_path)
batched_data = format_list_in_batch(data=golden_questions, batch_size=50)
golden_questions_batch_1 = batched_data[0]
eval_results = generate_rag_eval_result_with_retrieval_results(data=raw_search_results)
evaluation_results_gpt_4o_mini_path = project_root / "src" / "assets" / "evaluation_results_gpt_4o_mini.json"
save_json_file(data=eval_results, file_path=evaluation_results_gpt_4o_mini_path)
analyze_evaluation_result(file_path=evaluation_results_gpt_4o_mini_path)
gpt_4o_mini_eval_df = pd.DataFrame(data=eval_results)