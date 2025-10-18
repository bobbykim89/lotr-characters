from retrieval_evaluation import generate_questions_and_save_json, get_formatted_search_result, get_strategies_list, generate_evaluations_per_strategy, print_evaluation_result
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from composables.files import open_json_file, save_json_file
from composables.data_processing import format_list_in_batch

# process data and print retrieval evaluation from scratch

generate_questions_and_save_json()
golden_questions_file_path = project_root / "src" / "assets" / "golden_questions.json"
all_golden_questions = open_json_file(file_path=golden_questions_file_path)
batched_data = format_list_in_batch(data=all_golden_questions, batch_size=100)
golden_questions_batch_1 = batched_data[0]
search_results, last_index = get_formatted_search_result(golden_questions=golden_questions_batch_1)
retrieval_search_results_file_path = project_root / "src" / "assets" / "retrieval_search_results.json"
save_json_file(file_path=retrieval_search_results_file_path, data=search_results)
strategies = get_strategies_list()
eval_result = generate_evaluations_per_strategy(search_results=search_results, strategies=strategies)
print_evaluation_result(results=eval_result)