import json

def process_response_content(content: str):
    max_json_str = None
    max_len = 0
    for i, ch in enumerate(content):
        if ch == '{':
            counter = 0
            for j in range(i, len(content)):
                if content[j] == '{':
                    counter += 1
                elif content[j] == '}':
                    counter -= 1
                    if counter == 0:
                        candidate = content[i:j+1]
                        try:
                            json.loads(candidate)
                            if (j - i + 1) > max_len:
                                max_len = j - i + 1
                                max_json_str = candidate
                        except json.JSONDecodeError:
                            pass
                        break
    if max_json_str is not None:
        return json.loads(max_json_str)
    print("Raw response content:", content)
    raise ValueError("No valid JSON object found")