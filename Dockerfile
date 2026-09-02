# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL maintainer="Mostafa Khastkhodaei <dev.khastkhodaei@gmail.com>"
LABEL description="SemanticDFL Distributed Peer Node for EU SLICES Testbed"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and project configuration
COPY semantic_dfl/ /app/semantic_dfl/
COPY configs/ /app/configs/
COPY run_node.py /app/
COPY run_simulation.py /app/
COPY plot_results.py /app/

# Compile protobuf definitions in container
RUN python -m grpc_tools.protoc \
    -I /app/semantic_dfl/transport/proto \
    --python_out=/app/semantic_dfl/transport/grpc \
    --grpc_python_out=/app/semantic_dfl/transport/grpc \
    /app/semantic_dfl/transport/proto/dfl_transport.proto

# Create non-root user for testbed security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/results /app/Datasets && \
    chown -R appuser:appuser /app

USER appuser

# Expose default gRPC transport ports
EXPOSE 50050-50060

ENTRYPOINT ["python", "run_node.py"]
CMD ["--help"]
