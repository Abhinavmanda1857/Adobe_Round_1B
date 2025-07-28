FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Copy all code, models, and requirements
COPY requirements.txt .
COPY process_pdfs.py .
COPY docparser.py .
COPY doc_intelligence.py .
COPY doclayout_yolo/ ./doclayout_yolo/
COPY models/ ./models/
COPY outputs/ ./outputs/

RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/output && chmod -R 777 /app/output

# Run the script
CMD ["python", "process_pdfs.py"]
