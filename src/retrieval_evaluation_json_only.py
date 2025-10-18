import sys
from pathlib import Path
from retrieval_evaluation import get_strategies_list, generate_evaluations_per_strategy, print_evaluation_result

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from composables.files import open_json_file

# # Quickly process and print HIT RATE and MRR evaluation results based on saved retrieval_search_results saved in json file

retrieval_search_results_file_path = project_root / "src" / "assets" / "retrieval_search_results.json"
search_results = open_json_file(file_path=retrieval_search_results_file_path)
strategies = get_strategies_list()
eval_result = generate_evaluations_per_strategy(search_results=search_results, strategies=strategies)
print_evaluation_result(results=eval_result)