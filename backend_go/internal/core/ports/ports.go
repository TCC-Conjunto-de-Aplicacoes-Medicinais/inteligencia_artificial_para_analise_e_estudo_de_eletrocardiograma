package ports

import (
	"context"

	"github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/core/domain"
)

// PredictorPort define a interface de um agente especializado de inferência ONNX.
type PredictorPort interface {
	AgentName() string
	Predict(ctx context.Context, signal []float32) (float32, string, error)
}

// AnalyticsRepositoryPort define a interface de persistência assíncrona no Apache Cassandra.
type AnalyticsRepositoryPort interface {
	SaveAnalysis(ctx context.Context, result *domain.AnalysisResult) error
}

// OrchestratorPort define a interface de caso de uso principal para orquestração dos agentes de IA.
type OrchestratorPort interface {
	ProcessAnalysis(ctx context.Context, payload *domain.ECGSignalPayload) (*domain.AnalysisResult, error)
}
