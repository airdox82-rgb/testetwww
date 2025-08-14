FROM python:3.11-slim

WORKDIR /app
COPY app.py /app
RUN pip install flask flask-cors

EXPOSE 9871
CMD ["python", "app.py"]
