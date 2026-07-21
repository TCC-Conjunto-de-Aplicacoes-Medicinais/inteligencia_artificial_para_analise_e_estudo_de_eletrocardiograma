# 📝 TODO List — Módulo de IA Multi-Agente para Análise de ECG (Open Health SaaS)

Este documento centraliza as tarefas necessárias para estruturar o microserviço de IA, preparar os pipelines de treinamento em Python, construir o motor de inferência em Go com arquitetura Hexagonal e integrar o runtime ONNX.

---

## 🎯 Fase 1: Estruturação Inicial do Projeto & Arquitetura Hexagonal em Go

- [x] **1.1. Configuração do ambiente e dependências do Go**
  - [x] Inicializar o módulo Go (`go.mod`) na pasta `backend_go/`.
  - [x] Configurar pacotes essenciais (`gin` para REST, `gocql` para Cassandra, `onnxruntime-go` para CGO).
- [x] **1.2. Camada de Domínio (`internal/core/domain`)**
  - [x] Criar `ecg_signal.go`: Estrutura do payload de sinal ECG (`tenant_id`, `patient_id`, `sampling_rate`, vetores `[]float32`).
  - [x] Criar `analysis_result.go`: Structs do resultado diagnósticos, probabilidades e score de confiança.
  - [x] Criar `errors.go`: Erros de domínio estruturados (`ErrCorruptedSignal`, `ErrInvalidTenant`, `ErrONNXInferenceFailed`).
- [x] **1.3. Camada de Portas (`internal/core/ports`)**
  - [x] Definir `predictor_port.go`: Interface dos agentes de inferência ONNX (`Predict(ctx, vector) (score, string, error)`).
  - [x] Definir `repository_port.go`: Interface de persistência no Apache Cassandra.
  - [x] Definir `orchestrator_port.go`: Interface da regra de negócio de análise multi-agente.
- [x] **1.4. Camada de Serviços (`internal/core/services`)**
  - [x] Implementar `orchestrator_service.go`: Regra de negócio que executa os dois agentes ONNX via `goroutines` e seleciona o diagnóstico de maior confiança probabilística.
- [x] **1.5. Camada de Adaptadores (`internal/adapters`)**
  - [x] Implementar `adapters/http/ecg_handler.go`: Endpoint `POST /api/v1/ecg/analyze` com suporte a multitenancy (`tenant_id`).
  - [x] Implementar `adapters/onnx/onnx_engine.go`: Wrapper do runtime `onnxruntime-go` para carregamento dos modelos na memória.
  - [x] Implementar `adapters/storage/cassandra_repository.go`: Persistência assíncrona dos exames.
- [x] **1.6. Servidor e Containerização**
  - [x] Criar `cmd/server/main.go` para injeção de dependências e inicialização do servidor HTTP.
  - [x] Criar `backend_go/Dockerfile` configurado para compilar CGO com a biblioteca compartilhada ONNX (`libonnxruntime.so`).
  - [x] Criar `docker-compose.yml` para orquestração local (Go Backend + Cassandra).

---

## 🐍 Fase 2: Pipeline de Treinamento em Python & Preparação do Dataset

- [x] **2.1. Estruturação do repositório de ML (`ml_pipeline/`)**
  - [x] Criar `requirements.txt` (`torch`, `tsai`, `wfdb`, `scikit-learn`, `onnx`).
  - [x] Script de pré-processamento (`ml_pipeline/data/preprocess.py`): Normalização Z-Score e amostragem fixa (1000 pontos por lead).
- [x] **2.2. Construção das Redes 1D-CNN / ResNet-1D (`ml_pipeline/models/`)**
  - [x] Implementar arquitetura de rede neural convolucional 1D (`resnet1d.py`) para séries temporais.
- [x] **2.3. Script de Exportação ONNX (`ml_pipeline/export/export_onnx.py`)**
  - [x] Implementar exportação de PyTorch `.pt` para `.onnx` usando `torch.onnx.export()`.
- [x] **2.4. Geração de Modelo Dummy / Stub para Testes Integrados**
  - [x] Gerar script `ml_pipeline/export_dummy_onnx.py` para criar `agente_infarto.onnx` e `agente_sobrecarga.onnx` de teste imediatamente.

---

## 🚀 Fase 3: Treinamento dos Agentes de IA em Nuvem (Google Colab / GPU)

- [ ] **3.1. Treino do Agente 1 (Infarto Agudo do Miocárdio - IAM)**
  - [ ] Rodar `train_infarto.py` no Google Colab com GPU T4 e dataset PTB-XL.
  - [ ] Gerar gráficos de curva AUC-ROC para a monografia do TCC.
  - [ ] Exportar arquivo final `agente_infarto.onnx`.
- [ ] **3.2. Treino do Agente 2 (Sobrecarga / Condução / Arritmia)**
  - [ ] Rodar `train_sobrecarga.py` no Google Colab com GPU T4 e dataset PTB-XL.
  - [ ] Gerar gráficos de curva AUC-ROC para a monografia do TCC.
  - [ ] Exportar arquivo final `agente_sobrecarga.onnx`.

---

## 🧪 Fase 4: Testes de Carga & Validação de Desempenho em Produção

- [ ] **4.1. Teste de Inferência via Container Docker**
  - [ ] Subir ambiente local via `docker-compose up --build`.
  - [ ] Validar a requisição `POST /api/v1/ecg/analyze` com payloads de diferentes clínicas (`tenant_id`).
  - [ ] Verificar tempo de execução da inferência (meta: < 10ms por exame).
- [ ] **4.2. Persistência no Apache Cassandra**
  - [ ] Verificar inserção dos históricos de análise na tabela `ecg_analytics.diagnostic_results`.
