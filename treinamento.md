# 🧠 Guia Completo de Treinamento dos Agentes de IA em ECG (PyTorch & Google Colab)

Este documento detalha o processo de treinamento das redes neurais convolucionais unidimensionais (**1D-CNN / ResNet-1D**), o pré-processamento de sinal de voltagem de ECG e o passo a passo para executar o treinamento na GPU gratuita do **Google Colab**, exportando os modelos treinados para o formato **.onnx**.

---

## 📌 1. Visão Geral da Metodologia e Datasets

### Por que usar 1D-CNN em Séries Temporais de ECG?
O Eletrocardiograma (ECG) é um sinal biológico unidimensional de alta densidade temporal. Redes Convolucionais Unidimensionais (1D-CNN / ResNet-1D) são capazes de extrair automaticamente padrões morfológicos locais (como elevações de segmento ST, alterações de onda T, complexos QRS alargados e intervalos PR/QT) sem depender de engenharia manual de características (*feature engineering*).

### Datasets Recomendados:
1. **PTB-XL (PhysioNet)**:
   - **Tamanho**: 21.837 exames clínicos de ECG de 12 derivações de 18.885 pacientes.
   - **Rotulagem**: Padrão SCP-ECG cobrindo diagnósticos de Infarto (MI), Alterações de Condução (CD), Sobrecarga Ventricular (HYP) e Ritmo Sinusal Normal (NORM).
2. **MIT-BIH Arrhythmia Database**:
   - Dataset de referência para arritmias e batimentos ectópicos.

---

## ⚙️ 2. Pré-processamento e Arquitetura do Modelo

### Pré-processamento do Sinal
Cada derivação (lead) de ECG passa pelo seguinte tratamento no script `ml_pipeline/data/preprocess.py`:
1. **Tratativa de Ruídos**: Substituição de valores inválidos (`NaN`/`Inf`) por $0.0$.
2. **Amostragem Fixa**: Truncamento ou *zero-padding* para garantir um vetor fixo de **1000 pontos de voltagem**.
3. **Normalização Z-Score**:
   $$\hat{X} = \frac{X - \mu}{\sigma}$$

### Arquitetura ResNet-1D (`ml_pipeline/models/resnet1d.py`)
- **Camada Inicial**: `Conv1d(1, 32, kernel_size=15, stride=2, padding=7)` + `BatchNorm1d` + `ReLU` + `MaxPool1d`.
- **Blocos Residuais 1D**: Três blocos com conexões de atalho (*skip-connections*) incrementando canais ($32 \to 64 \to 128$).
- **Pooling & Classificação**: `AdaptiveAvgPool1d(1)` + `Linear(128, 2)` + `Softmax`.

---

## 🚀 3. Passo a Passo de Treinamento no Google Colab (GPU Gratuita)

Como o treinamento com 22 mil registros é otimizado para GPU, recomendava-se utilizar o **Google Colab** (demora **menos de 15 minutos por agente** na GPU T4).

### **Passo 1: Acessar o Google Colab**
1. Acesse [Google Colab](https://colab.research.google.com/).
2. Crie um **Novo Notebook** (`Notebook sem título`).
3. Vá no menu superior em **Ambiente de execução** $\to$ **Alterar tipo de ambiente de execução**.
4. Selecione **GPU T4** na opção *Acelerador de hardware* e clique em **Salvar**.

---

### **Passo 2: Clonar o Repositório e Instalar Dependências**
No primeiro bloco de código do Colab, execute:

```python
# 1. Clonar o repositório do projeto
!git clone https://github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma.git
%cd inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma

# 2. Instalar dependências de ML
!pip install -r ml_pipeline/requirements.txt
```

---

### **Passo 2.5: Baixar o Dataset PTB-XL (Necessário para Treino Real)**
Para treinar as redes com dados reais de ECG em vez de dados simulados (stubs/dummy), você precisa baixar e descompactar o dataset PTB-XL (cerca de 1.7 GB) na pasta raiz do projeto. Execute o seguinte bloco de código no Colab:

```python
# Garantir que estamos na pasta raiz do repositório
%cd /content/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma

# Baixar o dataset oficial PTB-XL do PhysioNet
!wget -O ptb-xl.zip https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip

# Descompactar na pasta atual
!unzip -q ptb-xl.zip
```

---

### **Passo 3: Executar os Scripts de Treinamento e Exportação ONNX**

> ⚠️ **Nota:** Como você reiniciou o ambiente de execução no passo anterior, o Google Colab reseta a pasta atual de volta para `/content`. Certifique-se de navegar de volta para a pasta do repositório antes de rodar os scripts.

#### Treinar o Agente 1 (Infarto Agudo do Miocárdio - IAM):
```python
%cd /content/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma
!python -m ml_pipeline.train_infarto
```

#### Treinar o Agente 2 (Sobrecarga / Condução / Arritmia):
```python
%cd /content/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma
!python -m ml_pipeline.train_sobrecarga
```

> 💡 **O que estes scripts fazem automaticamente?**
> 1. Carregam os sinais de ECG e rotulagens.
> 2. Treinam as redes 1D-CNN por 5 a 10 épocas na GPU.
> 3. Imprimem as métricas de validação (**AUC-ROC**, F1-Score e Acurácia).
> 4. Exportam automaticamente os modelos para os arquivos binários `.onnx`:
>    - `backend_go/models_onnx/agente_infarto.onnx`
>    - `backend_go/models_onnx/agente_sobrecarga.onnx`

---

### **Passo 4: Baixar os Arquivos `.onnx` para o Projeto**
Após a conclusão dos scripts no Colab, faça o download dos arquivos binários gerados para o seu computador:

```python
from google.colab import files

# Baixar os modelos exportados
files.download('backend_go/models_onnx/agente_infarto.onnx')
files.download('backend_go/models_onnx/agente_sobrecarga.onnx')
```

Copie os arquivos baixados para a pasta local do seu projeto em:
`backend_go/models_onnx/agente_infarto.onnx`
`backend_go/models_onnx/agente_sobrecarga.onnx`

---

## 📊 4. Guardando Gráficos de Validação (AUC-ROC) para o TCC

Durante o treinamento, o script salvará as curvas **AUC-ROC** e as matrizes de confusão. Guarde essas imagens na pasta `docs/` do repositório para inclusão direta no texto da monografia do TCC:

- `docs/auc_roc_infarto.png`
- `docs/auc_roc_sobrecarga.png`

Exemplo de valor de referência esperado na literatura com PTB-XL + ResNet-1D:
- **AUC-ROC (IAM)**: $\ge 0.92$
- **AUC-ROC (Sobrecarga/Arritmias)**: $\ge 0.89$
