package storage

import (
	"context"
	"fmt"
	"log"

	"github.com/gocql/gocql"
	"github.com/pohinc/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/core/domain"
)

type CassandraRepository struct {
	session *gocql.Session
}

func NewCassandraRepository(hosts []string, keyspace string) (*CassandraRepository, error) {
	cluster := gocql.NewCluster(hosts...)
	cluster.Keyspace = keyspace
	cluster.Consistency = gocql.Quorum

	session, err := cluster.CreateSession()
	if err != nil {
		log.Printf("[WARNING] Apache Cassandra não conectado (%v). Operando com fallback local.", err)
		return &CassandraRepository{session: nil}, nil
	}

	return &CassandraRepository{session: session}, nil
}

func (r *CassandraRepository) SaveAnalysis(ctx context.Context, result *domain.AnalysisResult) error {
	if r.session == nil {
		// Log analítico em stdout se Cassandra estiver em fallback
		log.Printf("[ANALYTICS FALLBACK] Tenant: %s | Paciente: %s | Diagnóstico: %s | Confiança: %.2f",
			result.TenantID, result.PatientID, result.PrimaryDiagnosis, result.HighestConfidence)
		return nil
	}

	query := `INSERT INTO ecg_analytics.diagnostic_results (
		analysis_id, tenant_id, patient_id, primary_diagnosis, winning_agent, highest_confidence, execution_time_ms, created_at
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`

	err := r.session.Query(query,
		result.AnalysisID,
		result.TenantID,
		result.PatientID,
		result.PrimaryDiagnosis,
		result.WinningAgent,
		result.HighestConfidence,
		result.ExecutionTimeMs,
		result.Timestamp,
	).WithContext(ctx).Exec()

	if err != nil {
		return fmt.Errorf("erro ao inserir no Apache Cassandra: %w", err)
	}

	return nil
}

func (r *CassandraRepository) Close() {
	if r.session != nil {
		r.session.Close()
	}
}
