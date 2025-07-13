#!/usr/bin/env python3
"""
Enhanced Fraud Detection API with Versioning and CI/CD
=====================================================

Implementation of feedback suggestions:
1. Model versioning system
2. CI/CD hooks and automated deployment
3. Request logging store (S3/DB)
4. Advanced monitoring and observability
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import joblib
import logging
import json
import asyncio
import aiofiles
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import os
import sqlite3
import boto3
from botocore.exceptions import NoCredentialsError
import uvicorn
import time
from contextlib import asynccontextmanager

# Configuration
MODEL_VERSION_PATH = "model_versions"
REQUEST_LOG_DB = "request_logs.db"
S3_BUCKET = os.getenv("FRAUD_API_S3_BUCKET", "fraud-detection-logs")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionRequest(BaseModel):
    """Request model for fraud prediction."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., ge=0, description="Transaction amount")
    time: float = Field(..., ge=0, description="Time from first transaction")
    features: Dict[str, float] = Field(..., description="V1-V28 features")
    merchant_category: Optional[str] = Field(None, description="Merchant category")
    location: Optional[str] = Field(None, description="Transaction location")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_12345",
                "amount": 100.50,
                "time": 3600.0,
                "features": {
                    "V1": -1.359807134,
                    "V2": -0.072781173,
                    "V3": 2.536346738,
                    "V4": 1.378155224
                },
                "merchant_category": "grocery",
                "location": "US"
            }
        }

class BatchTransactionRequest(BaseModel):
    """Request model for batch predictions."""
    transactions: List[TransactionRequest]
    
class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    model_version: str
    processing_time_ms: float
    explanation: Optional[Dict[str, Any]] = None
    
class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float
    batch_id: str

class ModelVersion(BaseModel):
    """Model version information."""
    version: str
    model_hash: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    is_active: bool
    description: str

class ModelVersionManager:
    """
    Manages model versions with automatic rollback and deployment tracking.
    
    Implements feedback suggestion for model versioning.
    """
    
    def __init__(self, base_path: str = MODEL_VERSION_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.versions_file = self.base_path / "versions.json"
        self.active_version = None
        self.models = {}
        
        self._load_versions()
        self._load_active_model()
    
    def _load_versions(self):
        """Load version metadata."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def _save_versions(self):
        """Save version metadata."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model file."""
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def deploy_model(self, model_data: Dict, version: str, description: str = "",
                    performance_metrics: Dict[str, float] = None) -> bool:
        """
        Deploy a new model version.
        
        Args:
            model_data: Dictionary containing models, scaler, etc.
            version: Version string (e.g., "v1.0.0")
            description: Human-readable description
            performance_metrics: Model performance metrics
        
        Returns:
            Success status
        """
        try:
            logger.info(f"Deploying model version {version}")
            
            # Save model files
            version_path = self.base_path / version
            version_path.mkdir(exist_ok=True)
            
            model_file = version_path / "model.joblib"
            joblib.dump(model_data, model_file)
            
            # Calculate hash
            model_hash = self._calculate_model_hash(model_file)
            
            # Update version metadata
            self.versions[version] = {
                "version": version,
                "model_hash": model_hash,
                "created_at": datetime.now().isoformat(),
                "performance_metrics": performance_metrics or {},
                "is_active": False,
                "description": description,
                "model_file": str(model_file)
            }
            
            self._save_versions()
            logger.info(f"Model version {version} deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model version {version}: {e}")
            return False
    
    def activate_version(self, version: str) -> bool:
        """
        Activate a specific model version.
        
        Args:
            version: Version to activate
        
        Returns:
            Success status
        """
        try:
            if version not in self.versions:
                logger.error(f"Version {version} not found")
                return False
            
            # Deactivate current version
            if self.active_version:
                self.versions[self.active_version]["is_active"] = False
            
            # Activate new version
            self.versions[version]["is_active"] = True
            self.active_version = version
            
            # Load model
            model_file = self.versions[version]["model_file"]
            self.models[version] = joblib.load(model_file)
            
            self._save_versions()
            logger.info(f"Activated model version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate version {version}: {e}")
            return False
    
    def _load_active_model(self):
        """Load the currently active model."""
        for version, metadata in self.versions.items():
            if metadata.get("is_active", False):
                self.active_version = version
                try:
                    model_file = metadata["model_file"]
                    self.models[version] = joblib.load(model_file)
                    logger.info(f"Loaded active model version {version}")
                except Exception as e:
                    logger.error(f"Failed to load active model {version}: {e}")
                break
    
    def get_active_model(self):
        """Get the currently active model."""
        if self.active_version and self.active_version in self.models:
            return self.models[self.active_version], self.active_version
        return None, None
    
    def list_versions(self) -> List[ModelVersion]:
        """List all model versions."""
        versions = []
        for version, metadata in self.versions.items():
            versions.append(ModelVersion(
                version=metadata["version"],
                model_hash=metadata["model_hash"],
                created_at=datetime.fromisoformat(metadata["created_at"]),
                performance_metrics=metadata["performance_metrics"],
                is_active=metadata["is_active"],
                description=metadata["description"]
            ))
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def rollback_to_previous(self) -> bool:
        """
        Rollback to the previous stable version.
        
        Returns:
            Success status
        """
        try:
            versions = self.list_versions()
            if len(versions) < 2:
                logger.warning("No previous version available for rollback")
                return False
            
            # Find previous version (not currently active)
            previous_version = None
            for version in versions[1:]:  # Skip current (first in sorted list)
                if not version.is_active:
                    previous_version = version.version
                    break
            
            if previous_version:
                logger.info(f"Rolling back to version {previous_version}")
                return self.activate_version(previous_version)
            else:
                logger.warning("No previous version found for rollback")
                return False
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

class RequestLogger:
    """
    Logs requests to database and optionally S3.
    
    Implements feedback suggestion for request logging store.
    """
    
    def __init__(self, db_path: str = REQUEST_LOG_DB, s3_enabled: bool = True):
        self.db_path = db_path
        self.s3_enabled = s3_enabled
        self.s3_client = None
        
        if s3_enabled:
            try:
                self.s3_client = boto3.client('s3', region_name=AWS_REGION)
                logger.info("S3 client initialized for request logging")
            except NoCredentialsError:
                logger.warning("AWS credentials not found, S3 logging disabled")
                self.s3_enabled = False
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for request logging."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT UNIQUE NOT NULL,
                    transaction_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    endpoint TEXT,
                    method TEXT,
                    model_version TEXT,
                    request_data TEXT,
                    response_data TEXT,
                    processing_time_ms REAL,
                    client_ip TEXT,
                    user_agent TEXT,
                    status_code INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON request_logs(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_transaction_id ON request_logs(transaction_id)
            """)
    
    async def log_request(self, request: Request, transaction_id: str, 
                         model_version: str, request_data: Dict,
                         response_data: Dict, processing_time_ms: float,
                         status_code: int):
        """
        Log request to database and S3.
        
        Args:
            request: FastAPI request object
            transaction_id: Transaction ID
            model_version: Model version used
            request_data: Request payload
            response_data: Response payload
            processing_time_ms: Processing time
            status_code: HTTP status code
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Log to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO request_logs 
                    (request_id, transaction_id, timestamp, endpoint, method, 
                     model_version, request_data, response_data, processing_time_ms,
                     client_ip, user_agent, status_code)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request_id,
                    transaction_id,
                    timestamp,
                    str(request.url.path),
                    request.method,
                    model_version,
                    json.dumps(request_data),
                    json.dumps(response_data),
                    processing_time_ms,
                    request.client.host if request.client else "unknown",
                    request.headers.get("user-agent", "unknown"),
                    status_code
                ))
        except Exception as e:
            logger.error(f"Failed to log to database: {e}")
        
        # Log to S3 (async)
        if self.s3_enabled:
            asyncio.create_task(self._log_to_s3(request_id, timestamp, {
                "request_id": request_id,
                "transaction_id": transaction_id,
                "timestamp": timestamp.isoformat(),
                "endpoint": str(request.url.path),
                "method": request.method,
                "model_version": model_version,
                "request_data": request_data,
                "response_data": response_data,
                "processing_time_ms": processing_time_ms,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "status_code": status_code
            }))
    
    async def _log_to_s3(self, request_id: str, timestamp: datetime, log_data: Dict):
        """Log request data to S3."""
        try:
            key = f"fraud-api-logs/{timestamp.strftime('%Y/%m/%d')}/{request_id}.json"
            
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=json.dumps(log_data, default=str),
                ContentType="application/json"
            )
            
        except Exception as e:
            logger.error(f"Failed to log to S3: {e}")
    
    def get_request_logs(self, limit: int = 100, transaction_id: str = None) -> List[Dict]:
        """
        Retrieve request logs from database.
        
        Args:
            limit: Maximum number of logs to return
            transaction_id: Filter by transaction ID
        
        Returns:
            List of log entries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if transaction_id:
                    cursor = conn.execute("""
                        SELECT * FROM request_logs 
                        WHERE transaction_id = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (transaction_id, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM request_logs 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to retrieve logs: {e}")
            return []

class FraudPredictor:
    """Enhanced fraud predictor with model versioning."""
    
    def __init__(self, version_manager: ModelVersionManager):
        self.version_manager = version_manager
    
    def predict(self, transaction: TransactionRequest) -> Dict:
        """
        Make fraud prediction for a single transaction.
        
        Args:
            transaction: Transaction data
        
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        # Get active model
        model_data, version = self.version_manager.get_active_model()
        
        if not model_data:
            raise HTTPException(status_code=503, detail="No active model available")
        
        try:
            # Prepare features
            features = self._prepare_features(transaction)
            
            # Get models
            models = model_data.get('models', {})
            scaler = model_data.get('scaler')
            
            if not models:
                raise HTTPException(status_code=503, detail="Model data invalid")
            
            # Use best performing model (assume Random Forest for demo)
            model_name = list(models.keys())[0]  # First available model
            model = models[model_name]
            
            # Scale features if scaler available
            if scaler:
                features_scaled = scaler.transform([features])
            else:
                features_scaled = [features]
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                fraud_probability = model.predict_proba(features_scaled)[0][1]
            else:
                fraud_probability = float(prediction)
            
            # Calculate risk score (0-100)
            risk_score = min(100, fraud_probability * 100)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Generate explanation (simplified)
            explanation = self._generate_explanation(transaction, fraud_probability)
            
            return {
                "transaction_id": transaction.transaction_id,
                "is_fraud": bool(prediction),
                "fraud_probability": float(fraud_probability),
                "risk_score": float(risk_score),
                "model_version": version,
                "processing_time_ms": processing_time,
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _prepare_features(self, transaction: TransactionRequest) -> List[float]:
        """Prepare feature vector from transaction."""
        # Start with Amount and Time
        features = [transaction.amount, transaction.time]
        
        # Add V features (V1-V28)
        for i in range(1, 29):
            v_key = f"V{i}"
            features.append(transaction.features.get(v_key, 0.0))
        
        return features
    
    def _generate_explanation(self, transaction: TransactionRequest, 
                            fraud_probability: float) -> Dict:
        """Generate simple explanation for the prediction."""
        explanation = {
            "risk_factors": [],
            "protective_factors": [],
            "confidence": "high" if abs(fraud_probability - 0.5) > 0.3 else "medium"
        }
        
        # Simple rule-based explanations
        if transaction.amount > 1000:
            explanation["risk_factors"].append("High transaction amount")
        elif transaction.amount < 10:
            explanation["risk_factors"].append("Very low transaction amount")
        
        if transaction.amount < 100:
            explanation["protective_factors"].append("Normal transaction amount")
        
        # Add feature-based explanations (simplified)
        v_features = [transaction.features.get(f"V{i}", 0) for i in range(1, 5)]
        if any(abs(v) > 3 for v in v_features):
            explanation["risk_factors"].append("Unusual transaction pattern detected")
        
        return explanation

# Initialize components
model_version_manager = ModelVersionManager()
request_logger = RequestLogger()
fraud_predictor = FraudPredictor(model_version_manager)

# FastAPI app with lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Enhanced Fraud Detection API")
    
    # Load default model if available
    try:
        default_models = joblib.load('fraud_models.joblib')
        default_scaler = joblib.load('scaler.joblib')
        default_results = joblib.load('model_results.joblib')
        
        model_data = {
            'models': default_models,
            'scaler': default_scaler,
            'results': default_results
        }
        
        # Deploy as initial version if no versions exist
        if not model_version_manager.versions:
            model_version_manager.deploy_model(
                model_data, 
                "v1.0.0", 
                "Initial model deployment",
                default_results.get(list(default_results.keys())[0], {})
            )
            model_version_manager.activate_version("v1.0.0")
            logger.info("Deployed and activated initial model version v1.0.0")
        
    except Exception as e:
        logger.warning(f"Could not load default models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Fraud Detection API")

app = FastAPI(
    title="Enhanced Fraud Detection API",
    description="Advanced fraud detection with model versioning and comprehensive logging",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# API Endpoints

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Predict fraud for a single transaction."""
    start_time = time.time()
    
    try:
        # Make prediction
        result = fraud_predictor.predict(transaction)
        
        # Log request
        background_tasks.add_task(
            request_logger.log_request,
            request,
            transaction.transaction_id,
            result["model_version"],
            transaction.dict(),
            result,
            result["processing_time_ms"],
            200
        )
        
        return PredictionResponse(**result)
        
    except HTTPException as e:
        # Log error
        background_tasks.add_task(
            request_logger.log_request,
            request,
            transaction.transaction_id,
            "unknown",
            transaction.dict(),
            {"error": str(e.detail)},
            (time.time() - start_time) * 1000,
            e.status_code
        )
        raise e
    except Exception as e:
        # Log unexpected error
        background_tasks.add_task(
            request_logger.log_request,
            request,
            transaction.transaction_id,
            "unknown",
            transaction.dict(),
            {"error": str(e)},
            (time.time() - start_time) * 1000,
            500
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    batch_request: BatchTransactionRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Predict fraud for multiple transactions."""
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    try:
        predictions = []
        
        for transaction in batch_request.transactions:
            result = fraud_predictor.predict(transaction)
            predictions.append(PredictionResponse(**result))
        
        total_time = (time.time() - start_time) * 1000
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time,
            batch_id=batch_id
        )
        
        # Log batch request
        background_tasks.add_task(
            request_logger.log_request,
            request,
            batch_id,
            "batch",
            {"batch_size": len(batch_request.transactions)},
            {"batch_id": batch_id, "predictions_count": len(predictions)},
            total_time,
            200
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/versions", response_model=List[ModelVersion])
async def list_model_versions():
    """List all model versions."""
    return model_version_manager.list_versions()

@app.post("/api/v1/models/activate/{version}")
async def activate_model_version(version: str):
    """Activate a specific model version."""
    success = model_version_manager.activate_version(version)
    
    if success:
        return {"message": f"Model version {version} activated successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to activate version {version}")

@app.post("/api/v1/models/rollback")
async def rollback_model():
    """Rollback to previous model version."""
    success = model_version_manager.rollback_to_previous()
    
    if success:
        return {"message": "Model rolled back successfully"}
    else:
        raise HTTPException(status_code=400, detail="Rollback failed")

@app.get("/api/v1/logs")
async def get_request_logs(limit: int = 100, transaction_id: Optional[str] = None):
    """Get request logs."""
    logs = request_logger.get_request_logs(limit=limit, transaction_id=transaction_id)
    return {"logs": logs, "count": len(logs)}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    model_data, version = model_version_manager.get_active_model()
    
    return {
        "status": "healthy" if model_data else "degraded",
        "active_model_version": version,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time()
    }

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get API metrics."""
    # Get recent request stats
    recent_logs = request_logger.get_request_logs(limit=1000)
    
    total_requests = len(recent_logs)
    successful_requests = sum(1 for log in recent_logs if log['status_code'] == 200)
    avg_processing_time = np.mean([log['processing_time_ms'] for log in recent_logs]) if recent_logs else 0
    
    return {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        "average_processing_time_ms": avg_processing_time,
        "active_model_version": model_version_manager.active_version
    }

# CI/CD Webhook endpoint
@app.post("/api/v1/deploy")
async def deploy_model_webhook(deployment_data: Dict[str, Any]):
    """
    Webhook for automated model deployment from CI/CD.
    
    Implements feedback suggestion for CI/CD hooks.
    """
    try:
        version = deployment_data.get("version")
        model_url = deployment_data.get("model_url")
        description = deployment_data.get("description", "")
        performance_metrics = deployment_data.get("performance_metrics", {})
        
        if not version or not model_url:
            raise HTTPException(status_code=400, detail="Version and model_url required")
        
        # Download and deploy model (simplified for demo)
        # In production, this would download from artifact store
        logger.info(f"Deploying model {version} from {model_url}")
        
        # For demo, assume model is already available locally
        try:
            model_data = joblib.load('fraud_models.joblib')  # Placeholder
            success = model_version_manager.deploy_model(
                model_data, version, description, performance_metrics
            )
            
            if success:
                # Auto-activate if performance is better
                if performance_metrics.get("f1_score", 0) > 0.8:
                    model_version_manager.activate_version(version)
                    logger.info(f"Auto-activated high-performing model {version}")
                
                return {"message": f"Model {version} deployed successfully"}
            else:
                raise HTTPException(status_code=500, detail="Deployment failed")
                
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Deployment webhook failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )