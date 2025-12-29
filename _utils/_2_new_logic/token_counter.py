import json
import tiktoken
from pathlib import Path

# Input and output paths
INPUT_FILE = Path("data\\0_input\\messages.jsonl")
OUTPUT_FILE = Path("messages_with_tokens.jsonl")

# Choose encoding for OpenAI text-embedding-3-small
encoding = tiktoken.encoding_for_model("text-embedding-3-small")

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    if text is None:
        return 0
    return len(encoding.encode(text))

def process_file(input_file: Path, output_file: Path):
    with input_file.open("r", encoding="utf-8") as fin, \
         output_file.open("w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line")
                continue

            # Count tokens for the "text" field
            text_val = obj.get("text", "")
            obj["tokens"] = count_tokens(text_val)

            # Write updated JSONL
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Done! File saved to: {output_file}")

if __name__ == "__main__":
    process_file(INPUT_FILE, OUTPUT_FILE)
