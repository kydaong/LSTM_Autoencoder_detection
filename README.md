# LSTM_Autoencoder_detection
This project uses an LSTM Autoencoder to detect anomalies in time-series sensor data in Centrifugal Compressors.  The model is trained to reconstruct normal operating patterns — when the reconstruction error exceeds a learned threshold, the system flags an anomalous event.

A production-ready deep learning pipeline for real-time anomaly detection in industrial compressor systems using Compressor-Aware LSTM autoencoders.

## Overview Contents
* [Architecture](#Architecture)
* [Key Design Decisions](#KeyDesignDecisions)
* [Technical Challenges & Solutions](#technicalchallenges&solutions)
* [Installation](#installation)
* [Configuration](#Configuration)
* [Usage](#usage)
* [Pipeline Components](#pipelinecomponents)
* [Performance & Monitoring](#performance&monitoring)
* [Production Deployment](#productiondeployment)

## Introduction

This repository implements a Compressor-Aware LSTM Autoencoder system for detecting anomalies in industrial compressor operations. Unlike traditional anomaly detection approaches, this system:

Multi-device learning: Learns per-device anomaly patterns by conditioning the model on compressor identity
Per-device thresholds: Maintains device-specific anomaly thresholds for accurate detection across heterogeneous equipment
Production-grade pipeline: Designed for continuous operation in industrial IoT environments
Automated scheduling: Fully automated data ingestion, model inference, and results persistence

Core Problem
Industrial compressors exhibit unique behavioral patterns based on their operational history, configuration, and maintenance status. A single global anomaly threshold fails to capture these nuances, leading to:

High false positive rates on "normally operating but abnormal for this device" patterns
Missed true anomalies masked by device-specific variance
Inability to distinguish between equipment degradation and installation differences

Solution
The Compressor-Aware LSTM approach uses a selective multi-head decoder architecture where:

All compressors share a common encoder for efficient representation learning
Each compressor has a dedicated decoder head for device-specific reconstruction
A custom routing layer selects the appropriate decoder based on compressor ID
Per-device thresholds are calibrated during training using normal operation data

# Architecture
##### High-level System Design
```mermaid
graph TB
    subgraph DataLayer["📊 Data Ingestion Layer"]
        DB["Azure SQL Server<br/>100M+ rows<br/>compressor_normal_dataset3"]
    end
    style DataLayer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    style DB fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000

    
    subgraph Orchestration["🎯 Orchestration"]
        SCHEDULER["LSTM_Scheduler.py<br/>- Timing every 60min<br/>- Checkpoint management<br/>- New data detection<br/>- Error recovery"]
    end
    style Orchestration fill:#e3f2fd,stroke:#1976d2,stroke-width:1px,color:#000
    style SCHEDULER fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    
    subgraph Pipeline["🔄 Prediction Pipeline"]
        STEP1["STEP 1: Data Reading<br/>Query date range<br/>1000-2000 records/compressor"]
        STEP2["STEP 2: Preprocessing<br/>- StandardScaler normalize<br/>- LabelEncoder compressor ID<br/>- Create 30-step sequences<br/>- Validate data quality"]
        STEP3["STEP 3: Model Inference<br/>GPU/CPU forward pass<br/>~30ms per 128 sequences"]
        STEP4["STEP 4: Anomaly Scoring<br/>- Per-feature MSE<br/>- Threshold comparison<br/>- Consecutive trigger"]
        STEP5["STEP 5: Persistence<br/>Write to 3 tables<br/>Update checkpoint"]
    end
    classDef nodestyle fill:#e3f2fd,stroke:#0d47a1,color:#000
    class STEP1,STEP2,STEP3,STEP4,STEP5 nodestyle
    
    subgraph Results["💾 Results Storage"]
        TABLE1["lstm_predictions_detailed<br/>100K rows per run"]
        TABLE2["lstm_batch_summary<br/>Aggregated stats"]
        TABLE3["anomaly_detection_results<br/>Simple format"]
        CHECKPOINT["lstm_scheduler_checkpoint<br/>Timestamp tracking"]
    end
    class TABLE1,TABLE2,TABLE3,CHECKPOINT nodestyle

    
    DB -->|Query| SCHEDULER
    SCHEDULER -->|Trigger| STEP1
    STEP1 -->|Raw data| STEP2
    STEP2 -->|Sequences| STEP3
    STEP3 -->|Reconstructions| STEP4
    STEP4 -->|Results| STEP5
    STEP5 --> TABLE1 & TABLE2 & TABLE3 & CHECKPOINT
    CHECKPOINT -->|Next checkpoint| SCHEDULER
    
    style DB fill:#e1f5ff
    style SCHEDULER fill:#fff3e0
    style STEP1 fill:#f3e5f5
    style STEP2 fill:#f3e5f5
    style STEP3 fill:#f3e5f5
    style STEP4 fill:#f3e5f5
    style STEP5 fill:#f3e5f5
    style TABLE1 fill:#e8f5e9
    style TABLE2 fill:#e8f5e9
    style TABLE3 fill:#e8f5e9
    style CHECKPOINT fill:#fce4ec


```

### Model Architecture Details
    INPUT: (batch, 30, 11)
        ▼
    ┌──────────────────────────────────────────┐
    │ ENCODER - Shared                         │
    ├──────────────────────────────────────────┤
    │ • LSTM(64)  → (batch, 30, 64)            │
    │ • LSTM(32)  → (batch, 32)                │
    │ • Dense(16) → (batch, 16)                │
    └──────────────────────────────────────────┘
        ▼
    ┌──────────────────────────────────────────┐
    │ DECODER HEADS - Per-Compressor           │
    ├──────────────────────────────────────────┤
    │ For each compressor:                     │
    │ • RepeatVector(30) → (batch, 30, 16)     │
    │ • LSTM(32)  → (batch, 30, 32)            │
    │ • LSTM(64)  → (batch, 30, 64)            │
    │ • Dense(11) → (batch, 30, 11)            │
    └──────────────────────────────────────────┘
        ▼
    ┌──────────────────────────────────────────┐
    │ SELECTOR - Route by Compressor ID        │
    ├──────────────────────────────────────────┤
    │ SelectDecoderOutput (Custom Layer)       │
    └──────────────────────────────────────────┘
        ▼
    OUTPUT: (batch, 30, 11)



This custom layer enables:

Single forward pass without branching logic in training/inference
TensorFlow graph optimization and XLA compilation support
Model serialization/deserialization without Lambda layers

## Feature Engineering
### Input Features (11 total):
1. filter_dp - Filter differential pressure
2. seal_gas_flow - Seal gas volumetric flow
3. seal_gas_diff_pressure - Seal gas pressure differential
4. seal_gas_temp - Seal gas temperature
5. primary_vent_flow - Primary vent flow rate
6. primary_vent_pressure - Primary vent pressure
7. secondary_seal_gas_flow - Secondary seal flow
8. separation_seal_gas_flow - Separation gas flow
9. separation_seal_gas_pressure - Separation gas pressure
10. seal_gas_to_vent_diff_pressure - Gas-to-vent pressure differential
11. encoding - [One-hot or label encoded compressor ID]

Dimensionality:

Sequence length: 30 timesteps (15 minutes at 30-second intervals)
Total input shape: (batch_size, 30, 11)


# Key Design Decisions 

1. Compressor-Aware vs. Global Model

| Aspect | Global Model | Compressor-Aware |
|--------|:----------:|:----------:|
| Threshold | Single value | Per-device calibrated |
| False positives | High (device variance) | Low (learned variance) |
| Training time | Faster | Moderate (multi-decoder) |
| Deployment | Simple | Requires encoder mapping |
| New compressor | Retrain full model | Add new decoder + threshold |