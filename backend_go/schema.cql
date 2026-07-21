-- Script de inicialização do Cassandra para persistência analítica de ECG (Multi-Tenant)

-- 1. Criar o Keyspace
CREATE KEYSPACE IF NOT EXISTS ecg_analytics 
WITH replication = {
    'class': 'SimpleStrategy', 
    'replication_factor': 1
};

USE ecg_analytics;

-- 2. Criar a Tabela de Resultados de Diagnóstico
CREATE TABLE IF NOT EXISTS diagnostic_results (
    analysis_id uuid,
    tenant_id text,
    patient_id text,
    primary_diagnosis text,
    winning_agent text,
    highest_confidence float,
    execution_time_ms int,
    created_at timestamp,
    PRIMARY KEY (tenant_id, created_at, analysis_id)
) WITH CLUSTERING ORDER BY (created_at DESC, analysis_id ASC);
