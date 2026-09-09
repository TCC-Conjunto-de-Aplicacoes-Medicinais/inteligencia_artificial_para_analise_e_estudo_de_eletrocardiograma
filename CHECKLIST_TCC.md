# 📋 Checklist de Alinhamento Acadêmico — CONIC 2026
## Módulo 4: Pipeline de Inteligência Artificial & Microserviço de Inferência em ECG

> **Documento Base:** *Artigo CONIC SEMESP 2026 — Plataforma SaaS Open Health (POHINC)*  
> **Repositório:** `inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma`  
> **Tecnologias Centrais:** PyTorch, tsai, 1D-CNN (ResNet-1D), Go (Gin), ONNX Runtime (`onnxruntime-go`), Apache Cassandra

---

### 📌 1. Visão Geral do Módulo no Artigo
Conforme descrito nas Seções **1, 3.2, 4.3, 5, 6.1 e 7** do artigo:
* **Papel:** Arquitetura multiagente paralela de Redes Neurais Convolucionais Unidimensionais (1D-CNN) para triagem de eletrocardiogramas (ECG) como segunda opinião médica.
* **Agentes Especialistas:**
  1. **Agente 1 (Infarto):** Calibrado para elevações de segmento ST e alterações isquêmicas agudas.
  2. **Agente 2 (Sobrecarga Ventricular):** Calibrado para hipertrofias cavitárias e distúrbios de condução.
* **Métricas Experimentais Publicadas no Artigo:**
  - **Dataset:** PTB-XL canônico (21.801 exames de 12 derivações a 100 Hz, 1000 pontos por derivação).
  - **Validação Cruzada:** 10 folds estratificados (*strat_fold*, sem vazamento de pacientes).
  - **Sensibilidade / Taxa de Acerto:** **82,4%** (superando a meta de $\ge 80\%$).
  - **Capacidade Discriminatória (AUC-ROC):** **0,86** (na faixa-alvo de $0,82$ a $0,88$).
  - **Tempo de Inferência:** Na escala de milissegundos via `onnxruntime-go` em memória.

---

### 🎯 2. Status Atual da Implementação
- [x] Backend em Go com Arquitetura Hexagonal (Portas e Adaptadores).
- [x] Orquestrador multiagente paralelo via *goroutines* com seleção por consenso estatístico (`highest_confidence`).
- [x] Tabela do Cassandra definida em `backend_go/schema.cql` (`ecg_analytics.diagnostic_results`).
- [x] Pipeline em Python com ResNet-1D (`resnet1d.py`), pré-processamento Z-Score (`preprocess.py`), scripts de treino `train_infarto.py` e `train_sobrecarga.py`, e exportador dummy `export_dummy_onnx.py`.

---

### ⏳ 3. Pendências e Itens Faltantes para Alinhamento com a Documentação

#### 3.1. Geração e Disponibilização dos Modelos ONNX — 🚨 GAP CRÍTICO
- [ ] **Criar a pasta `backend_go/models_onnx/` e gerar os binários ONNX:**
  - *Problema:* Os arquivos `agente_infarto.onnx` e `agente_sobrecarga.onnx` **não existem** no repositório.
  - *Consequência:* O backend em Go ativa automaticamente o modo de fallback (`useFallback = true`), gerando resultados fictícios e mockados em vez de inferência de rede neural real.
  - *Ação Imediata (Ambiente Local/Testes):*
    Executar o script gerador de modelos dummy:
    ```bash
    python -m ml_pipeline.export_dummy_onnx
    ```
    Isso gerará os arquivos `.onnx` válidos na pasta `backend_go/models_onnx/`.
  - *Ação Definitiva (Apresentação do Artigo):*
    Executar o treinamento no Google Colab seguindo o guia [treinamento.md](file:///c:/Users/iisai/OneDrive/Desktop/TCC/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/treinamento.md), baixar os modelos treinados com as métricas publicadas (AUC-ROC de 0,86) e salvá-los no repositório.

#### 3.2. Integração do Microserviço com os Demais Módulos
- [ ] **Conectar o Endpoint `POST /api/v1/ecg/analyze`:**
  - *Problema:* O microsserviço de IA está completamente isolado dos demais repositórios.
  - *Ação:*
    1. Conectar a tela do prontuário médico das clínicas (`motor_para_geracao_de_sistemas_clinicos`) para despachar o arquivo de ECG para esta API.
    2. Disponibilizar a chamada a partir do Barramento Centralizador para exames enviados pelo aplicativo móvel do paciente.

#### 3.3. Camada Textual Futura (MedGemma via RAG)
- [ ] **Definição de Interfaces Arquiteturais para LLM Clínico:**
  - O artigo cita na Seção 4.3: *"Em nível arquitetural, projetou-se a conexão com modelos de linguagem clínica assistidos por RAG para processamento de narrativas médicas livres. A integração com MedGemma/RAG permanece configurada para síntese textual complementar"*.
  - *Ação:* Criar na camada de portas (`backend_go/internal/core/ports/`) uma interface Go para síntese textual (ex: `TextSynthesizerPort`) com um adaptador stub preparado para receber chaves do Google Gemini / MedGemma.
