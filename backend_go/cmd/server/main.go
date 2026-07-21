package main

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
	httpadapter "github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/adapters/http"
	onnxadapter "github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/adapters/onnx"
	storageadapter "github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/adapters/storage"
	"github.com/TCC-Conjunto-de-Aplicacoes-Medicinais/inteligencia_artificial_para_analise_e_estudo_de_eletrocardiograma/backend_go/internal/core/services"
)

func main() {
	log.Println("🫀 Inicializando Microserviço de IA para Análise de ECG (Multi-Tenant)...")

	// 1. Tenta inicializar o runtime ONNX CGO
	onnxLibPath := os.Getenv("ONNX_SHARED_LIB_PATH")
	if err := onnxadapter.InitONNXRuntime(onnxLibPath); err != nil {
		log.Printf("[INFO] ONNX CGO inicializado com modo fallback/stub de desenvolvimento: %v", err)
	}

	// 2. Inicializa os Preditores Especialistas ONNX
	agent1Path := getEnvOrDefault("AGENT_INFARTO_ONNX", "models_onnx/agente_infarto.onnx")
	agent2Path := getEnvOrDefault("AGENT_SOBRECARGA_ONNX", "models_onnx/agente_sobrecarga.onnx")

	agentIAM, err := onnxadapter.NewONNXAgentPredictor("Agente 1 - Infarto Agudo do Miocárdio", agent1Path)
	if err != nil {
		log.Fatalf("Erro ao carregar Agente IAM: %v", err)
	}
	defer agentIAM.Close()

	agentSobrecarga, err := onnxadapter.NewONNXAgentPredictor("Agente 2 - Sobrecarga/Arritmia", agent2Path)
	if err != nil {
		log.Fatalf("Erro ao carregar Agente Sobrecarga: %v", err)
	}
	defer agentSobrecarga.Close()

	// 3. Inicializa o Repositório Apache Cassandra (com fallback local se offline)
	cassandraHosts := []string{getEnvOrDefault("CASSANDRA_HOST", "127.0.0.1")}
	cassandraKeyspace := getEnvOrDefault("CASSANDRA_KEYSPACE", "ecg_analytics")
	cassandraRepo, _ := storageadapter.NewCassandraRepository(cassandraHosts, cassandraKeyspace)
	defer cassandraRepo.Close()

	// 4. Instancia o Serviço Orquestrador
	orchestrator := services.NewOrchestratorService(
		cassandraRepo,
		agentIAM,
		agentSobrecarga,
	)

	// 5. Configura o Servidor HTTP REST (Gin)
	gin.SetMode(getEnvOrDefault("GIN_MODE", gin.ReleaseMode))
	router := gin.Default()

	ecgHandler := httpadapter.NewECGHandler(orchestrator)
	ecgHandler.RegisterRoutes(router)

	port := getEnvOrDefault("PORT", "8080")
	log.Printf("🚀 Servidor de IA operando na porta %s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Erro ao iniciar servidor HTTP: %v", err)
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultValue
}
