# syntax=docker/dockerfile:1

FROM python:3.13.2-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py"]