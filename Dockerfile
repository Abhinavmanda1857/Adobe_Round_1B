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

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/output /app/models && chmod -R 777 /app/output

# Download model file into models directory
RUN curl -L https://huggingface.co/Adieee5/adobe_model_parse/resolve/main/last.pt -o /app/models/last.pt

# Run the script
CMD ["python", "process_pdfs.py"]
