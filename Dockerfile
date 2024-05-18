
FROM python:3.10


WORKDIR /app

COPY application /app

RUN pip install uv
RUN uv venv

# RUN pip install --no-cache-dir -r /app/requirements.txt

RUN pip install -r /app/requirements.txt


EXPOSE 5001
EXPOSE 6333
EXPOSE 6334
EXPOSE 8501


CMD ["python", "/app/app.py"]
