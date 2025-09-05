# Modelado y Simulación 1

Este repositorio contiene una aplicación Streamlit para cálculo de raíces de ecuaciones.

## Requisitos

- Docker
- Docker Compose

## Instrucciones de Instalación y Ejecución

1. Clona este repositorio:
```bash
git clone https://github.com/baridie/modelado_simulacion_1.git
cd modelado_simulacion_1
```

2. Construye y levanta el contenedor usando Docker Compose:
```bash
docker compose build
docker compose up
```

3. Una vez que el contenedor esté corriendo, abre tu navegador y visita:
```
http://localhost:8501
```

La aplicación Streamlit estará disponible y lista para usar.

## Detener la Aplicación

Para detener la aplicación, puedes presionar `Ctrl+C` en la terminal donde está corriendo el contenedor, o ejecutar en otra terminal:
```bash
docker compose down