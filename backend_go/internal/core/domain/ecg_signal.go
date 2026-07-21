package domain

import "time"

// ECGSignalPayload representa o payload recebido do SaaS principal contendo os vetores de voltagem.
type ECGSignalPayload struct {
	TenantID          string               `json:"tenant_id" binding:"required"`
	PatientID         string               `json:"patient_id" binding:"required"`
	ExamID            string               `json:"exam_id,omitempty"`
	SamplingRateHz    int                  `json:"sampling_rate_hz" binding:"required"`
	RequestedAnalyses []string             `json:"requested_analyses,omitempty"` // Ex: ["infarto", "sobrecarga"]
	Leads             map[string][]float32 `json:"leads" binding:"required"`             // Ex: {"lead_II": [0.012, 0.045, ...]}
	CreatedAt         time.Time            `json:"created_at,omitempty"`
}

// ECGVector representa a matriz processada pronta para ser alimentada no tensor do ONNX Runtime.
type ECGVector struct {
	TenantID  string
	PatientID string
	Data      []float32 // Vetor 1D contínuo de amostragem padronizada (ex: 1000 pontos)
	NumPoints int
}
