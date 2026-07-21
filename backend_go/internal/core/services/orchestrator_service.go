package services

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/core/domain"
	"github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/core/ports"
)

const TargetSignalLength = 1000

// OrchestratorService implementa a interface ports.OrchestratorPort com concorrência em Goroutines.
type OrchestratorService struct {
	predictors []ports.PredictorPort
	repo       ports.AnalyticsRepositoryPort
}

// NewOrchestratorService instancia o serviço orquestrador injetando os preditores ONNX e o repositório.
func NewOrchestratorService(repo ports.AnalyticsRepositoryPort, predictors ...ports.PredictorPort) *OrchestratorService {
	return &OrchestratorService{
		predictors: predictors,
		repo:       repo,
	}
}

// ProcessAnalysis realiza a normalização do vetor e orquestra a execução concorrente dos agentes ONNX.
func (s *OrchestratorService) ProcessAnalysis(ctx context.Context, payload *domain.ECGSignalPayload) (*domain.AnalysisResult, error) {
	startTime := time.Now()

	if payload.TenantID == "" {
		return nil, domain.ErrInvalidTenant
	}

	// Extrai e normaliza a derivação principal (ex: lead_II ou primeira derivação encontrada)
	rawVector, err := s.extractAndNormalizeLead(payload)
	if err != nil {
		return nil, err
	}

	type agentOutput struct {
		name       string
		score      float32
		diagnosis  string
		err        error
	}

	ch := make(chan agentOutput, len(s.predictors))
	var wg sync.WaitGroup

	// Execução paralela de cada modelo especialista ONNX
	for _, predictor := range s.predictors {
		wg.Add(1)
		go func(p ports.PredictorPort) {
			defer wg.Done()
			score, diag, pErr := p.Predict(ctx, rawVector)
			ch <- agentOutput{
				name:      p.AgentName(),
				score:     score,
				diagnosis: diag,
				err:       pErr,
			}
		}(predictor)
	}

	wg.Wait()
	close(ch)

	detailedScores := make(map[string]domain.AgentScore)
	var highestConfidence float32 = -1.0
	var winningAgent string
	var primaryDiagnosis string

	for out := range ch {
		if out.err != nil {
			return nil, fmt.Errorf("falha no agente ONNX [%s]: %w", out.name, out.err)
		}

		detailedScores[out.name] = domain.AgentScore{
			AgentName:  out.name,
			Diagnosis:  out.diagnosis,
			Confidence: out.score,
		}

		if out.score > highestConfidence {
			highestConfidence = out.score
			winningAgent = out.name
			primaryDiagnosis = out.diagnosis
		}
	}

	result := &domain.AnalysisResult{
		AnalysisID:        uuid.New().String(),
		TenantID:          payload.TenantID,
		PatientID:         payload.PatientID,
		PrimaryDiagnosis:  primaryDiagnosis,
		WinningAgent:      winningAgent,
		HighestConfidence: highestConfidence,
		DetailedScores:    detailedScores,
		ProcessedPoints:   len(rawVector),
		ExecutionTimeMs:   time.Since(startTime).Milliseconds(),
		Timestamp:         time.Now(),
	}

	// Persistência assíncrona no Apache Cassandra (não bloqueia a resposta REST rápida)
	if s.repo != nil {
		go func(res *domain.AnalysisResult) {
			_ = s.repo.SaveAnalysis(context.Background(), res)
		}(result)
	}

	return result, nil
}

// extractAndNormalizeLead garante que o sinal tenha exatamente TargetSignalLength (1000) elementos e esteja normalizado.
func (s *OrchestratorService) extractAndNormalizeLead(payload *domain.ECGSignalPayload) ([]float32, error) {
	if len(payload.Leads) == 0 {
		return nil, domain.ErrCorruptedSignal
	}

	// Tenta pegar "lead_II", caso contrário pega o primeiro lead disponível
	var raw []float32
	if l, ok := payload.Leads["lead_II"]; ok && len(l) > 0 {
		raw = l
	} else {
		for _, l := range payload.Leads {
			if len(l) > 0 {
				raw = l
				break
			}
		}
	}

	if len(raw) == 0 {
		return nil, domain.ErrCorruptedSignal
	}

	// Validação de NaN e Infinity
	for _, val := range raw {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			return nil, domain.ErrCorruptedSignal
		}
	}

	// Resampling / Zero-Padding para ajustar o tamanho para 1000 pontos
	normalized := make([]float32, TargetSignalLength)
	if len(raw) >= TargetSignalLength {
		copy(normalized, raw[:TargetSignalLength])
	} else {
		copy(normalized, raw)
		// Zero padding implícito pois slice recém-criado possui 0.0
	}

	// Normalização Z-Score Standard Scaling
	var sum float64
	for _, v := range normalized {
		sum += float64(v)
	}
	mean := sum / float64(TargetSignalLength)

	var varianceSum float64
	for _, v := range normalized {
		diff := float64(v) - mean
		varianceSum += diff * diff
	}
	stdDev := math.Sqrt(varianceSum / float64(TargetSignalLength))
	if stdDev == 0 {
		stdDev = 1.0 // Previne divisão por zero
	}

	for i := 0; i < TargetSignalLength; i++ {
		normalized[i] = float32((float64(normalized[i]) - mean) / stdDev)
	}

	return normalized, nil
}
