FROM python:3.11-slim

WORKDIR /workspace

RUN pip install --no-cache-dir pandas numpy

COPY run_code.py /workspace/run_code.py

CMD ["python", "/workspace/run_code.py"]