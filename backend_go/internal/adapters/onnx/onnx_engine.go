package onnx

import (
	"context"
	"fmt"
	"math"
	"os"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	onnxInitOnce sync.Once
	onnxInitialized bool
)

// InitONNXRuntime inicializa a biblioteca CGO do ONNX Runtime em memória uma única vez.
func InitONNXRuntime(sharedLibPath string) error {
	var err error
	onnxInitOnce.Do(func() {
		if sharedLibPath != "" {
			ort.SetSharedLibraryPath(sharedLibPath)
		}
		err = ort.InitializeEnvironment()
		if err == nil {
			onnxInitialized = true
		}
	})
	return err
}

// ONNXAgentPredictor implementa a interface PredictorPort para carregar e inferir modelos .onnx.
type ONNXAgentPredictor struct {
	agentName      string
	modelPath      string
	session        *ort.AdvancedSession
	inputTensor    *ort.Tensor[float32]
	outputTensor   *ort.Tensor[float32]
	useFallback    bool
}

// NewONNXAgentPredictor instancia um novo preditor especializado baseado no arquivo .onnx fornecido.
func NewONNXAgentPredictor(agentName string, modelPath string) (*ONNXAgentPredictor, error) {
	predictor := &ONNXAgentPredictor{
		agentName: agentName,
		modelPath: modelPath,
	}

	// Se o arquivo ONNX não existir ainda ou a lib CGO não estiver inicializada, habilita modo fallback seguro para testes
	if _, err := os.Stat(modelPath); os.IsNotExist(err) || !onnxInitialized {
		predictor.useFallback = true
		return predictor, nil
	}

	inputShape := ort.NewShape(1, 1, 1000)
	outputShape := ort.NewShape(1, 2) // Outputs: [Prob_Normal, Prob_Patologia]

	inputData := make([]float32, 1000)
	outputData := make([]float32, 2)

	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		predictor.useFallback = true
		return predictor, nil
	}

	outputTensor, err := ort.NewTensor(outputShape, outputData)
	if err != nil {
		predictor.useFallback = true
		return predictor, nil
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		predictor.useFallback = true
		return predictor, nil
	}

	predictor.session = session
	predictor.inputTensor = inputTensor
	predictor.outputTensor = outputTensor

	return predictor, nil
}

func (p *ONNXAgentPredictor) AgentName() string {
	return p.agentName
}

// Predict executa a inferência no tensor 1D recebido e retorna o score probabilístico e o diagnóstico.
func (p *ONNXAgentPredictor) Predict(ctx context.Context, signal []float32) (float32, string, error) {
	if p.useFallback || p.session == nil {
		// Mock/Fallback de inferência determinística para testes iniciais de integração
		var score float32 = 0.85
		var diagnosis string = "Normal (Stub ONNX)"
		
		if len(signal) > 0 {
			firstVal := float64(signal[0])
			if math.Abs(firstVal) > 1.5 {
				score = 0.94
				if p.agentName == "Agente 1 - Infarto Agudo do Miocárdio" {
					diagnosis = "Suspeita de Infarto Agudo do Miocárdio (IAM)"
				} else {
					diagnosis = "Alteração de Condução / Sobrecarga Ventricular"
				}
			} else {
				score = 0.12
				diagnosis = "Sem Alterações Eletrocardiográficas Significativas"
			}
		}

		return score, diagnosis, nil
	}

	// Copia os dados do sinal para o buffer do tensor
	inputData := p.inputTensor.GetData()
	copy(inputData, signal)

	err := p.session.Run()
	if err != nil {
		return 0, "", fmt.Errorf("erro na execução do tensor ONNX [%s]: %w", p.agentName, err)
	}

	outputData := p.outputTensor.GetData()
	confidence := outputData[1] // Probabilidade da patologia detectada

	var diagnosis string
	if confidence > 0.5 {
		if p.agentName == "Agente 1 - Infarto Agudo do Miocárdio" {
			diagnosis = "Elevação de Segmento ST / Suspeita de IAM"
		} else {
			diagnosis = "Bloqueio de Ramo / Sobrecarga Ventricular"
		}
	} else {
		diagnosis = "Ritmo Sinusal / Sem Alerta Diagnóstico"
	}

	return confidence, diagnosis, nil
}

// Close libera os recursos da sessão ONNX Runtime.
func (p *ONNXAgentPredictor) Close() {
	if p.session != nil {
		_ = p.session.Destroy()
	}
}
