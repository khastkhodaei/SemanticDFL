# syntax=docker/dockerfile:1
# Multi-Stage Production Dockerfile for SemanticDFL Peer Nodes
# Author: Mostafa Khastkhodaei <mostafa.khodaei@hafez.shirazu.ac.ir>

# ==============================================================================
# Stage 1: Build & Dependencies Compiler
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolated dependency packaging
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy protobuf definition and compile gRPC stubs
COPY semantic_dfl/transport/proto/dfl_transport.proto /build/proto/
RUN mkdir -p /build/compiled_grpc && \
    python -m grpc_tools.protoc \
        -I /build/proto \
        --python_out=/build/compiled_grpc \
        --grpc_python_out=/build/compiled_grpc \
        /build/proto/dfl_transport.proto

# ==============================================================================
# Stage 2: Minimal Production Runtime
# ==============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Mostafa Khastkhodaei <mostafa.khodaei@hafez.shirazu.ac.ir>"
LABEL description="SemanticDFL Multi-Modal Decentralized Federated Learning Node"
LABEL paper="Dogani, Khastkhodaei, Khunjush, Laoutaris (POMACS / ACM SIGMETRICS 2026)"

WORKDIR /app

# Unbuffered stdout/stderr for real-time per-container log streaming
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy compiled gRPC stubs
COPY --from=builder /build/compiled_grpc/ /app/semantic_dfl/transport/grpc/

# Copy application source code and configuration profiles
COPY semantic_dfl/ /app/semantic_dfl/
COPY configs/ /app/configs/
COPY run_node.py /app/
COPY run_simulation.py /app/
COPY plot_results.py /app/

# Setup volume mount points and permissions
RUN mkdir -p /app/results /app/Datasets /app/logs && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose range of gRPC transport ports for peer-to-peer communication
EXPOSE 50050-50100

ENTRYPOINT ["python", "run_node.py"]
CMD ["--help"]
