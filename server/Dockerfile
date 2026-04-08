FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "clinic_scheduler.server.app:app", "--host", "0.0.0.0", "--port", "8000"]