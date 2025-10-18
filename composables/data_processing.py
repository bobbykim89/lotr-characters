def format_list_in_batch(data: list, batch_size:int = 100)-> list[list]:
    total_entries = len(data)
    print(f"Total entries: {total_entries}")
    batch_idx = []
    for batch_num in range(0, total_entries, batch_size):
        batch_end = min(batch_num + batch_size, total_entries)
        batch_idx.append([batch_num, batch_end])
    
    formatted_data = []
    for batch in batch_idx:
        batched_data = data[batch[0]:batch[1]]
        formatted_data.append(batched_data)
        
    return formatted_data