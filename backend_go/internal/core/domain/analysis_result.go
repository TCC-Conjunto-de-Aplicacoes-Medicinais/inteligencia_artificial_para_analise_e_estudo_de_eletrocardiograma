package domain

import "time"

// AgentScore representa a saída estatística de um modelo especializado específico.
type AgentScore struct {
	AgentName  string  `json:"agent_name"`
	Diagnosis  string  `json:"diagnosis"`
	Confidence float32 `json:"confidence"`
}

// AnalysisResult representa o diagnóstico consolidado retornado pelo orquestrador de IA.
type AnalysisResult struct {
	AnalysisID        string                `json:"analysis_id"`
	TenantID          string                `json:"tenant_id"`
	PatientID         string                `json:"patient_id"`
	PrimaryDiagnosis  string                `json:"primary_diagnosis"`
	WinningAgent      string                `json:"winning_agent"`
	HighestConfidence float32               `json:"highest_confidence"`
	DetailedScores    map[string]AgentScore `json:"detailed_scores"`
	ProcessedPoints   int                   `json:"processed_points"`
	ExecutionTimeMs   int64                 `json:"execution_time_ms"`
	Timestamp         time.Time             `json:"timestamp"`
}
