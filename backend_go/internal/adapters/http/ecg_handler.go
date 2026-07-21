package http

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/core/domain"
	"github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/core/ports"
)

type ECGHandler struct {
	orchestrator ports.OrchestratorPort
}

func NewECGHandler(orchestrator ports.OrchestratorPort) *ECGHandler {
	return &ECGHandler{
		orchestrator: orchestrator,
	}
}

func (h *ECGHandler) RegisterRoutes(router *gin.Engine) {
	api := router.Group("/api/v1")
	{
		api.POST("/ecg/analyze", h.AnalyzeECG)
		api.GET("/health", h.HealthCheck)
	}
}

// AnalyzeECG cuida da recepção do vetor numérico de voltagem enviado pelos SaaS dos clientes/clínicas.
func (h *ECGHandler) AnalyzeECG(c *gin.Context) {
	var payload domain.ECGSignalPayload
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"status":  "error",
			"code":    "INVALID_PAYLOAD",
			"message": "Formato de payload inválido ou campos obrigatórios ausentes",
			"details": err.Error(),
		})
		return
	}

	// Permite injetar TenantID via Header HTTP se não vier no body
	if payload.TenantID == "" {
		payload.TenantID = c.GetHeader("X-Tenant-ID")
	}

	result, err := h.orchestrator.ProcessAnalysis(c.Request.Context(), &payload)
	if err != nil {
		switch err {
		case domain.ErrInvalidTenant:
			c.JSON(http.StatusUnauthorized, gin.H{"status": "error", "code": "INVALID_TENANT", "message": err.Error()})
		case domain.ErrCorruptedSignal:
			c.JSON(http.StatusUnprocessableEntity, gin.H{"status": "error", "code": "CORRUPTED_SIGNAL", "message": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"status": "error", "code": "INTERNAL_INFERENCE_ERROR", "message": err.Error()})
		}
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"data":   result,
	})
}

func (h *ECGHandler) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"service":   "ECG AI Microservice",
		"multitenant": true,
	})
}
