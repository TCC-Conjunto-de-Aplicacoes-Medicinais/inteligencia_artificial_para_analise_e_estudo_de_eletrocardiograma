import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ml_pipeline.models.resnet1d import ResNet1D
from ml_pipeline.export.export_onnx import export_model_to_onnx

def train_agent_sobrecarga(epochs=5, batch_size=32):
    print("🚀 Iniciando treinamento do Agente 2 - Sobrecarga Ventricular / Arritmias...")
    
    # Modelo 1D-CNN ResNet
    model = ResNet1D(in_channels=1, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Stub de dados para simulação de treino caso o dataset PTB-XL ainda não esteja baixado
    dummy_signals = torch.randn(100, 1, 1000)
    dummy_labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(dummy_signals, dummy_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")

    output_path = os.path.join(os.path.dirname(__file__), "..", "backend_go", "models_onnx", "agente_sobrecarga.onnx")
    export_model_to_onnx(model, output_path)

if __name__ == "__main__":
    train_agent_sobrecarga()
