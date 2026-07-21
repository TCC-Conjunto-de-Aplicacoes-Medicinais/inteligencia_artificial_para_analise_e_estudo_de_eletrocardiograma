import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

def generate_dummy_onnx():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "backend_go", "models_onnx")
    os.makedirs(output_dir, exist_ok=True)

    dummy_input = torch.randn(1, 1, 1000, dtype=torch.float32)

    # 1. Agente Infarto ONNX
    model_infarto = Simple1DCNN()
    model_infarto.eval()
    path_infarto = os.path.join(output_dir, "agente_infarto.onnx")
    torch.onnx.export(
        model_infarto, dummy_input, path_infarto,
        input_names=['input'], output_names=['output'],
        opset_version=14
    )
    print(f"📦 [STUB] agente_infarto.onnx gerado em: {path_infarto}")

    # 2. Agente Sobrecarga ONNX
    model_sobrecarga = Simple1DCNN()
    model_sobrecarga.eval()
    path_sobrecarga = os.path.join(output_dir, "agente_sobrecarga.onnx")
    torch.onnx.export(
        model_sobrecarga, dummy_input, path_sobrecarga,
        input_names=['input'], output_names=['output'],
        opset_version=14
    )
    print(f"📦 [STUB] agente_sobrecarga.onnx gerado em: {path_sobrecarga}")

if __name__ == "__main__":
    generate_dummy_onnx()
