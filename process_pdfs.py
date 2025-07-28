import os
from pathlib import Path
from docparser import DocParser

def process_pdfs():
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = DocParser(
        pdfs_dir=str(input_dir),
        outputs_dir=str(output_dir),
        model_path="/app/models/last.pt"
    )
    parser.process_all_pdfs()

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("completed processing pdfs")
