FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY hf_template.py .
COPY oai_template.py .

ADD test test/

CMD ["pytest", "--showlocals"]
