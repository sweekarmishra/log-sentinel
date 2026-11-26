FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
# The model is trained on first boot if artifacts/model.joblib is absent.
CMD gunicorn wsgi:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120
