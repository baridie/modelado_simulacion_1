FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY streamlit_raices_mejorado.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_raices_mejorado.py", "--server.address", "0.0.0.0"]