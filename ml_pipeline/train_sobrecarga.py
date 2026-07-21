import os
import ast
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from ml_pipeline.models.resnet1d import ResNet1D
from ml_pipeline.export.export_onnx import export_model_to_onnx

class PTBXLDataset(Dataset):
    def __init__(self, db_path, folds=[1,2,3,4,5,6,7,8], task='sobrecarga', sampling_rate=100):
        self.db_path = db_path
        self.sampling_rate = sampling_rate
        
        # Carrega metadados do dataset
        self.df = pd.read_csv(os.path.join(db_path, 'ptbxl_database.csv'), index_col='ecg_id')
        # Filtra os dados com base nos folds especificados
        self.df = self.df[self.df.strat_fold.isin(folds)]
        self.df['scp_codes'] = self.df['scp_codes'].apply(lambda x: ast.literal_eval(x))
        
        # Carrega mapeamento de diagnósticos scp
        self.df_scp = pd.read_csv(os.path.join(db_path, 'scp_statements.csv'), index_col=0)
        self.df_scp = self.df_scp[self.df_scp.diagnostic == 1]
        
        self.labels = []
        self.filenames = []
        
        for idx, row in self.df.iterrows():
            scp_dict = row['scp_codes']
            superclasses = []
            for code in scp_dict.keys():
                if code in self.df_scp.index:
                    superclass = self.df_scp.loc[code, 'diagnostic_class']
                    if pd.notna(superclass):
                        superclasses.append(superclass)
            
            # Agente 2 (Sobrecarga / Arritmia) foca em HYP (Hipertrofia) e CD (Distúrbio de Condução)
            label = 1 if ('HYP' in superclasses or 'CD' in superclasses) else 0
            self.labels.append(label)
            
            if sampling_rate == 100:
                self.filenames.append(row['filename_lr'])
            else:
                self.filenames.append(row['filename_hr'])
                
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.db_path, self.filenames[idx])
        # Carrega sinal ECG via wfdb
        signal, _ = wfdb.rdsamp(filename)
        
        # Seleciona Derivação II (índice 1)
        lead_signal = signal[:, 1]
        
        # Normalização Z-Score e Tratativa de NaNs
        lead_signal = np.nan_to_num(lead_signal, nan=0.0, posinf=0.0, neginf=0.0)
        mean = np.mean(lead_signal)
        std = np.std(lead_signal)
        if std == 0:
            std = 1.0
        normalized = (lead_signal - mean) / std
        
        # Garante tamanho fixo de 1000 pontos (Z-padding ou truncamento)
        target_len = 1000
        current_len = len(normalized)
        if current_len > target_len:
            start = (current_len - target_len) // 2
            normalized = normalized[start:start + target_len]
        elif current_len < target_len:
            pad_width = target_len - current_len
            normalized = np.pad(normalized, (0, pad_width), mode='constant')
            
        tensor_x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0) # [1, 1000]
        tensor_y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_x, tensor_y

def train_agent_sobrecarga(epochs=5, batch_size=64):
    print("🚀 Iniciando treinamento do Agente 2 - Sobrecarga Ventricular / Arritmias...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Utilizando dispositivo de hardware: {device}")
    
    # Modelo 1D-CNN ResNet
    model = ResNet1D(in_channels=1, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Caminho do dataset PTB-XL
    db_path = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    
    if os.path.exists(db_path):
        print("📦 Dataset PTB-XL encontrado! Carregando dados reais...")
        train_dataset = PTBXLDataset(db_path, folds=list(range(1, 9)), task='sobrecarga', sampling_rate=100)
        val_dataset = PTBXLDataset(db_path, folds=[9, 10], task='sobrecarga', sampling_rate=100)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        print("⚠️ Dataset PTB-XL não encontrado. Utilizando dados de simulação (Dummy/Stub)...")
        dummy_signals = torch.randn(200, 1, 1000)
        dummy_labels = torch.randint(0, 2, (200,))
        train_dataset = TensorDataset(dummy_signals, dummy_labels)
        val_dataset = TensorDataset(dummy_signals[:50], dummy_labels[:50])
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f} - Acurácia: {acc:.2f}%")

    # Avaliação de validação e geração da curva AUC-ROC
    print("📊 Iniciando validação e geração da curva AUC-ROC...")
    model.eval()
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            # A probabilidade da classe positiva (Sobrecarga = 1)
            probs = outputs[:, 1].cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(batch_y.numpy())
            
    # Calcular e plotar AUC-ROC
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    fpr, tpr, _ = roc_curve(val_targets, val_probs)
    roc_auc = auc(fpr, tpr)
    print(f"📈 AUC-ROC de Validação: {roc_auc:.4f}")
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo (FPR)')
    plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
    plt.title('Curva ROC - Agente Sobrecarga / Arritmias')
    plt.legend(loc="lower right")
    
    # Salvar em docs/auc_roc_sobrecarga.png
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, "auc_roc_sobrecarga.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Gráfico salvo com sucesso em: {plot_path}")

    output_path = os.path.join(os.path.dirname(__file__), "..", "backend_go", "models_onnx", "agente_sobrecarga.onnx")
    # Exportar modelo para CPU para inferência genérica posterior no Go
    export_model_to_onnx(model.to("cpu"), output_path)

if __name__ == "__main__":
    train_agent_sobrecarga()
