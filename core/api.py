from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from linear_regression import LinearRegression
from logistic import LogisticRegression
from svm import SVM
from neural_network import NeuralNetwork
from decision_tree import DecisionTree
from random_forest import RandomForest
from gradient_boosting import GradientBoosting

app = FastAPI(title="Optimizer Lens API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    algorithm: str
    X: List[List[float]]
    y: List[float]
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: Optional[int] = None
    # SVM parameters
    C: Optional[float] = 1.0
    kernel: Optional[str] = 'linear'
    gamma: Optional[float] = 0.1
    # Neural Network parameters
    hidden_layers: Optional[List[int]] = [64, 32]
    activation: Optional[str] = 'relu'
    # Tree-based parameters
    max_depth: Optional[int] = 10
    min_samples_split: Optional[int] = 2
    n_estimators: Optional[int] = 10

class TrainResponse(BaseModel):
    metrics: List[dict]
    weights: List[float]
    bias: float
    final_loss: float
    final_accuracy: float

class PredictRequest(BaseModel):
    algorithm: str
    X: List[List[float]]
    weights: List[float]
    bias: float

class PredictResponse(BaseModel):
    predictions: List[float]

@app.get("/")
def read_root():
    return {"message": "Optimizer Lens API", "status": "running"}

@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    try:
        X = np.array(request.X)
        y = np.array(request.y)
        
        metrics = []
        
        if request.algorithm == "linear":
            model = LinearRegression(
                learning_rate=request.learning_rate,
                iterations=request.epochs
            )
            
            # Track metrics during training
            temp_model = LinearRegression(
                learning_rate=request.learning_rate,
                iterations=1
            )
            temp_model.weights = np.zeros(X.shape[1])
            temp_model.bias = 0
            
            for epoch in range(request.epochs):
                temp_model.fit(X, y)
                y_pred = temp_model.predict(X)
                mse = LinearRegression.mean_sq_err(y, y_pred)
                
                # Calculate R² score as "accuracy" for regression
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                metrics.append({
                    "epoch": epoch + 1,
                    "loss": float(mse),
                    "accuracy": float(max(0, min(1, r2_score)))
                })
            
            model.fit(X, y)
            weights, bias = model.get_params()
            y_pred = model.predict(X)
            final_loss = float(LinearRegression.mean_sq_err(y, y_pred))
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            final_accuracy = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0
            
        elif request.algorithm == "logistic":
            model = LogisticRegression(
                learning_rate=request.learning_rate,
                iterations=request.epochs
            )
            
            # Track metrics during training
            temp_model = LogisticRegression(
                learning_rate=request.learning_rate,
                iterations=1
            )
            temp_model.weights = np.zeros(X.shape[1])
            temp_model.bias = 0
            
            for epoch in range(request.epochs):
                # Train for one epoch
                y_pred = temp_model._compute_prediction(X)
                dw, db = temp_model._compute_gradient(X, y, y_pred)
                temp_model._update_params(dw, db)
                
                # Calculate loss (binary cross-entropy)
                epsilon = 1e-15
                y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
                loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
                
                # Calculate accuracy
                predictions = temp_model.predict(X)
                accuracy = np.mean(predictions == y)
                
                metrics.append({
                    "epoch": epoch + 1,
                    "loss": float(loss),
                    "accuracy": float(accuracy)
                })
            
            model.fit(X, y)
            weights, bias = model.get_params()
            
            y_pred = model._compute_prediction(X)
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            final_loss = float(-np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped)))
            
            predictions = model.predict(X)
            final_accuracy = float(np.mean(predictions == y))
            
        elif request.algorithm == "svm" or request.algorithm == "kernel_svm":
            # Use kernel parameter from request
            kernel = request.kernel if request.algorithm == "kernel_svm" else 'linear'
            
            model = SVM(
                C=request.C,
                kernel=kernel,
                gamma=request.gamma,
                max_iter=min(request.epochs, 1000)  # Limit iterations for SVM
            )
            model.fit(X, y)
            
            # Generate metrics (SVM doesn't have iterative training like gradient descent)
            # We'll simulate progress
            for epoch in range(1, request.epochs + 1):
                progress = epoch / request.epochs
                
                # Simulate convergence
                y_pred = model.predict(X)
                accuracy = np.mean(y_pred == y)
                
                # Hinge loss approximation
                decision = model._decision_function(X)
                y_signed = np.where(y == 0, -1, 1)
                hinge_loss = np.mean(np.maximum(0, 1 - y_signed * decision))
                
                metrics.append({
                    "epoch": epoch,
                    "loss": float(hinge_loss),
                    "accuracy": float(accuracy * progress + 0.5 * (1 - progress))  # Simulate improvement
                })
            
            weights = [0.0]  # Placeholder
            bias = model.b
            
            y_pred = model.predict(X)
            final_accuracy = float(np.mean(y_pred == y))
            decision = model._decision_function(X)
            y_signed = np.where(y == 0, -1, 1)
            final_loss = float(np.mean(np.maximum(0, 1 - y_signed * decision)))
            
        elif request.algorithm == "neural_network":
            model = NeuralNetwork(
                hidden_layers=request.hidden_layers,
                activation=request.activation,
                learning_rate=request.learning_rate,
                epochs=request.epochs,
                batch_size=request.batch_size or 32
            )
            
            # Train and track metrics
            n_samples = X.shape[0]
            model._initialize_weights(X.shape[1])
            
            for epoch in range(request.epochs):
                # Forward and backward pass
                activations, z_values = model._forward_propagation(X)
                gradients_w, gradients_b = model._backward_propagation(X, y, activations, z_values)
                
                # Update weights
                for j in range(len(model.weights)):
                    model.weights[j] -= model.learning_rate * gradients_w[j]
                    model.biases[j] -= model.learning_rate * gradients_b[j]
                
                # Calculate metrics
                y_pred_proba = model.predict_proba(X)
                epsilon = 1e-15
                y_pred_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
                loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
                
                predictions = model.predict(X)
                accuracy = np.mean(predictions == y)
                
                metrics.append({
                    "epoch": epoch + 1,
                    "loss": float(loss),
                    "accuracy": float(accuracy)
                })
            
            weights = [0.0]  # Placeholder (network has multiple weight matrices)
            bias = 0.0
            final_loss = metrics[-1]["loss"]
            final_accuracy = metrics[-1]["accuracy"]
            
        elif request.algorithm == "decision_tree":
            model = DecisionTree(
                max_depth=request.max_depth,
                min_samples_split=request.min_samples_split
            )
            model.fit(X, y)
            
            # Decision trees don't have epochs, so we simulate progress
            predictions = model.predict(X)
            final_accuracy = float(np.mean(predictions == y))
            
            for epoch in range(1, request.epochs + 1):
                progress = min(1.0, epoch / 10)  # Quick convergence
                metrics.append({
                    "epoch": epoch,
                    "loss": float((1 - final_accuracy) * (1 - progress)),
                    "accuracy": float(final_accuracy * progress + 0.5 * (1 - progress))
                })
            
            weights = [0.0]
            bias = 0.0
            final_loss = metrics[-1]["loss"]
            
        elif request.algorithm == "random_forest":
            model = RandomForest(
                n_estimators=request.n_estimators,
                max_depth=request.max_depth,
                min_samples_split=request.min_samples_split
            )
            model.fit(X, y)
            
            predictions = model.predict(X)
            final_accuracy = float(np.mean(predictions == y))
            
            # Simulate progressive improvement as trees are added
            for epoch in range(1, request.epochs + 1):
                progress = min(1.0, epoch / 20)
                metrics.append({
                    "epoch": epoch,
                    "loss": float((1 - final_accuracy) * (1 - progress)),
                    "accuracy": float(final_accuracy * progress + 0.5 * (1 - progress))
                })
            
            weights = [0.0]
            bias = 0.0
            final_loss = metrics[-1]["loss"]
            
        elif request.algorithm == "gradient_boosting":
            model = GradientBoosting(
                n_estimators=min(request.n_estimators, 100),
                learning_rate=request.learning_rate,
                max_depth=request.max_depth
            )
            model.fit(X, y)
            
            predictions = model.predict(X)
            final_accuracy = float(np.mean(predictions == y))
            
            # Simulate progressive improvement
            for epoch in range(1, request.epochs + 1):
                progress = min(1.0, epoch / 30)
                metrics.append({
                    "epoch": epoch,
                    "loss": float((1 - final_accuracy) * (1 - progress) + 0.1),
                    "accuracy": float(final_accuracy * progress + 0.5 * (1 - progress))
                })
            
            weights = [0.0]
            bias = 0.0
            final_loss = metrics[-1]["loss"]
        
        else:
            raise HTTPException(status_code=400, detail=f"Algorithm '{request.algorithm}' not supported")
        
        return TrainResponse(
            metrics=metrics,
            weights=weights.tolist() if isinstance(weights, np.ndarray) else [weights] if isinstance(weights, (int, float)) else weights,
            bias=float(bias),
            final_loss=final_loss,
            final_accuracy=final_accuracy
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        X = np.array(request.X)
        weights = np.array(request.weights)
        bias = request.bias
        
        if request.algorithm == "linear":
            predictions = np.dot(X, weights) + bias
        elif request.algorithm == "logistic":
            z = np.dot(X, weights) + bias
            predictions = 1 / (1 + np.exp(-z))
        else:
            raise HTTPException(status_code=400, detail=f"Algorithm '{request.algorithm}' not supported")
        
        return PredictResponse(predictions=predictions.tolist())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
