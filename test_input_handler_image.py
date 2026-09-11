from pathlib import Path
from input_handler.handler import process_input

image_path = Path("architecture.png")

result = process_input(image_path)

print("\n==============================")
print("IMAGE INPUT HANDLER TEST")
print("==============================")

print("Input type:")
print(result["input_type"])

print("\nExtracted content:")
print("==============================")

print(result["extracted_content"])