FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "clinic_scheduler.server.app"]