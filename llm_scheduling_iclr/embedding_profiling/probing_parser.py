import re
import json


def parse_log(log_file_path):
    records = []
    current_record = {}
    iter_list = []
    capture_next_line_for_prompt = False

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    layers = []

    for i, line in enumerate(lines):
        if 'record~' in line:
            if current_record:
                current_record['iterations'] = iter_list
                records.append(current_record)
                iter_list = []
            current_record = {}
            record_match = re.search(r'record~(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{6}): (\d+)', line)
            if record_match:
                current_record['record_id'] = int(record_match.group(2))
                current_record['start_ts'] = record_match.group(1)

        elif 'Formatted prompt~' in line:
            prompt_match = re.search(r'Formatted prompt~\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{6}: (.+)', line)
            if prompt_match:
                current_record['record_prompt'] = prompt_match.group(1)
                capture_next_line_for_prompt = True
        elif capture_next_line_for_prompt:
            current_record['record_prompt'] += '\n\n' + line.strip()
            capture_next_line_for_prompt = False

        elif '**** Embedding from the ' in line:
            tensor_size_match = re.search(r'\*\*\*\* Embedding from the (\d+)th layer:.*torch\.Size\(\[(.*?)\]\)', line)
            timestamp_match = re.search(r'\*\*\*\* timestamp:  (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{6})',
                                        lines[i + 1])
            print(tensor_size_match, timestamp_match)
            if tensor_size_match and timestamp_match:
                layers.append({
                    'layer_id': int(tensor_size_match.group(1)),
                    'tensor_size': eval(tensor_size_match.group(2)),
                    'iter_ts': timestamp_match.group(1),
                    # 'file': f"L{tensor_size_match.group(1)}_{timestamp_match.group(1)}.pt"
                })
                if int(tensor_size_match.group(1)) == 15:
                    # print(f"Found all layers for record {current_record['record_id']}")
                    iter_list.append({
                        'iter_id': len(iter_list),
                        'layers': layers
                    })
                    layers = []
                # iter_list.append({
                #     'iter_id': len(iter_list),
                #     'tensor_size': eval(tensor_size_match.group(1)),
                #     'iter_ts': timestamp_match.group(1)
                # })

        elif 'Output~' in line:
            output_match = re.search(r'Output~(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d{6}): (.+)', line)
            if output_match:
                current_record['output'] = output_match.group(2)
                current_record['output_ts'] = output_match.group(1)
                current_record['iteration_count'] = len(iter_list)

    if current_record:
        current_record['iterations'] = iter_list
        records.append(current_record)

    return records


def write_json(records, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump({'records': records, 'total': len(records)}, json_file, indent=4)


file_name = 'nohup4k'

log_file_path = f'./{file_name}.out'
output_file_path = './llama3-4k.json'

records = parse_log(log_file_path)
write_json(records, output_file_path)