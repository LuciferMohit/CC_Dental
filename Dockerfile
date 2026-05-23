FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openpyxl mlflow torch torchvision scikit-learn pandas numpy matplotlib

EXPOSE 5000

CMD ["python", "src/evaluate.py"]
