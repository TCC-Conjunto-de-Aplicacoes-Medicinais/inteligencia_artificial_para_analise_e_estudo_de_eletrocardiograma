# Stage 1: Build
FROM golang:1.22-bullseye AS builder

WORKDIR /app

# Instala suporte C++ e CGO para a biblioteca ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Baixa a biblioteca compartilhada do ONNX Runtime para CGO
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.17.1.tgz \
    && cp onnxruntime-linux-x64-1.17.1/lib/libonnxruntime.so* /usr/lib/

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Compila o binário em Go habilitando CGO
ENV CGO_ENABLED=1
RUN go build -o server ./cmd/server

# Stage 2: Runtime
FROM debian:bullseye-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/lib/libonnxruntime.so* /usr/lib/
COPY --from=builder /app/server .
COPY --from=builder /app/models_onnx ./models_onnx

ENV PORT=8080
ENV ONNX_SHARED_LIB_PATH=/usr/lib/libonnxruntime.so

EXPOSE 8080

CMD ["./server"]
