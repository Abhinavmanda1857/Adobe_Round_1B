FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Copy all code, models, and requirements
COPY requirements.txt .
COPY process_pdfs.py .
COPY docparser.py .
COPY doclayout_yolo/ ./doclayout_yolo/
COPY models/ ./models/
COPY outputs/ ./outputs/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up default output folder and update permissions if necessary (optional)
RUN mkdir -p /app/output && chmod -R 777 /app/output

# Run the script
CMD ["python", "process_pdfs.py"]
