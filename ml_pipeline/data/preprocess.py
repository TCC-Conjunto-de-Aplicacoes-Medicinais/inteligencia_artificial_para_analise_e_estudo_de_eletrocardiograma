import numpy as np

TARGET_LENGTH = 1000

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normaliza uma série temporal 1D usando Z-Score Standard Scaling e
    garante dimensão fixa de TARGET_LENGTH (1000 pontos).
    """
    signal = np.array(signal, dtype=np.float32)
    
    # Tratativa para remoção de valores inválidos (NaN/Inf)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # Reamostragem / Zero-Padding para ajustar a 1000 pontos
    current_len = len(signal)
    if current_len > TARGET_LENGTH:
        # Truncamento centralizado
        start = (current_len - TARGET_LENGTH) // 2
        signal = signal[start:start + TARGET_LENGTH]
    elif current_len < TARGET_LENGTH:
        # Zero-padding ao final
        pad_width = TARGET_LENGTH - current_len
        signal = np.pad(signal, (0, pad_width), mode='constant', constant_values=0.0)

    # Z-Score Standard Scaling
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        std = 1.0

    normalized = (signal - mean) / std
    return normalized.astype(np.float32)
