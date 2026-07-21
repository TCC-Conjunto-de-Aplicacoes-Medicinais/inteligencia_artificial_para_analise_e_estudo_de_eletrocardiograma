package domain

import "errors"

var (
	// ErrInvalidTenant é retornado quando o tenant_id não é fornecido ou é inválido.
	ErrInvalidTenant = errors.New("tenant_id inválido ou ausente no cabeçalho/payload")

	// ErrCorruptedSignal é retornado quando o vetor numérico do ECG apresenta discrepâncias ou dados ausentes.
	ErrCorruptedSignal = errors.New("vetor de sinal ECG corrompido, vazio ou contendo valores inválidos (NaN/Inf)")

	// ErrInvalidSignalLength é retornado quando a amostragem do sinal não pode ser adequada.
	ErrInvalidSignalLength = errors.New("comprimento do vetor do sinal incompatível com a entrada do modelo")

	// ErrONNXInferenceFailed é retornado quando há falha na execução do tensor CGO no runtime ONNX.
	ErrONNXInferenceFailed = errors.New("falha na inferência dos agentes ONNX")
)
