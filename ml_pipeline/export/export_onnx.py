import os
import torch
from ml_pipeline.models.resnet1d import ResNet1D

def export_model_to_onnx(model: torch.nn.Module, output_path: str):
    """
    Exporta o modelo treinado em PyTorch para o formato interoperável ONNX.
    """
    model.eval()
    
    # Input dummy no formato [Batch_Size=1, Channels=1, Sequence_Length=1000]
    dummy_input = torch.randn(1, 1, 1000, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ Modelo ONNX exportado com sucesso para: {output_path}")

if __name__ == "__main__":
    model = ResNet1D()
    export_model_to_onnx(model, "../backend_go/models_onnx/agente_infarto.onnx")
