import os
from pathlib import Path
from docparser import DocParser
from doc_intelligence import DocumentIntelligence

def process_pdfs():
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    inter_output_dir = Path("/app/outputs")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = DocParser(
        pdfs_dir=str(input_dir),
        outputs_dir=str(inter_output_dir),
        model_path="/app/models/last.pt"
    )
    parser.process_all_pdfs()

    intelligence = DocumentIntelligence(
        input_dir=str(inter_output_dir),
        pdfs_dir=str(input_dir),
        output_json_path=str(output_dir)
    )

    result = intelligence.find_relevant_sections(
        persona="Travel Planner",
        job_to_be_done="Plan a trip of 4 days for a group of 10 college friends"
    )

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("completed processing pdfs")
