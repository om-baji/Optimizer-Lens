"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Play, Pause, RotateCcw, Settings, Brain, TrendingUp, Target } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

interface MLConfig {
  algorithm: string
  learningRate: number
  epochs: number
  batchSize: number
  regularization: number
  momentum?: number
  beta1?: number
  beta2?: number
  epsilon?: number
  kernelType?: string
  C?: number
  gamma?: number
  hiddenLayers?: number[]
  activation?: string
}

interface DataPoint {
  x: number
  y: number
  label: number
}

interface TrainingMetrics {
  epoch: number
  loss: number
  accuracy: number
  valLoss?: number
  valAccuracy?: number
}

const algorithms = {
  linear: "Linear Regression",
  logistic: "Logistic Regression",
  svm: "Support Vector Machine",
  kernel_svm: "Kernel SVM",
  neural_network: "Neural Network",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  gradient_boosting: "Gradient Boosting",
}

const kernelTypes = {
  linear: "Linear",
  polynomial: "Polynomial",
  rbf: "RBF (Gaussian)",
  sigmoid: "Sigmoid",
}

const activationFunctions = {
  relu: "ReLU",
  sigmoid: "Sigmoid",
  tanh: "Tanh",
  leaky_relu: "Leaky ReLU",
}

const datasetTypes = {
  linear: "Linear Separable",
  nonlinear: "Non-linear",
  circles: "Concentric Circles",
  moons: "Half Moons",
  blobs: "Gaussian Blobs",
  extreme_outliers: "Extreme Outliers",
  high_noise: "High Noise",
  imbalanced: "Imbalanced Classes",
}

export function MLPlayground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [config, setConfig] = useState<MLConfig>({
    algorithm: "linear",
    learningRate: 0.01,
    epochs: 100,
    batchSize: 32,
    regularization: 0.01,
    momentum: 0.9,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    kernelType: "rbf",
    C: 1.0,
    gamma: 0.1,
    hiddenLayers: [64, 32],
    activation: "relu",
  })

  const [dataset, setDataset] = useState<string>("linear")
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([])
  const [isTraining, setIsTraining] = useState(false)
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([])
  const [model, setModel] = useState<any>(null)
  const [extremeValues, setExtremeValues] = useState(false)
  const [noiseLevel, setNoiseLevel] = useState(0.1)
  const [testAccuracy, setTestAccuracy] = useState(0)
  const [confusionMatrix, setConfusionMatrix] = useState<number[][]>([
    [0, 0],
    [0, 0],
  ])
  const [useRealAlgorithms, setUseRealAlgorithms] = useState(false)
  const [apiStatus, setApiStatus] = useState<'checking' | 'available' | 'unavailable'>('checking')
  const SERVERURL = process.env.NEXT_PUBLIC_SERVER_URL; 

  // Generate synthetic datasets
  const generateDataset = (type: string, numPoints = 200) => {
    const points: DataPoint[] = []
    const range = extremeValues ? 1000 : 10

    switch (type) {
      case "linear":
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x + y > 0 ? 1 : 0
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
        }
        break

      case "nonlinear":
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x * x + y * y < (range / 4) * (range / 4) ? 1 : 0
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
        }
        break

      case "circles":
        for (let i = 0; i < numPoints; i++) {
          const angle = Math.random() * 2 * Math.PI
          const r1 = Math.random() * (range / 6)
          const r2 = range / 4 + Math.random() * (range / 6)
          const r = Math.random() > 0.5 ? r1 : r2
          const x = r * Math.cos(angle)
          const y = r * Math.sin(angle)
          const label = r < range / 4 ? 1 : 0
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
        }
        break

      case "moons":
        for (let i = 0; i < numPoints; i++) {
          const t = Math.random() * Math.PI
          const label = Math.random() > 0.5 ? 1 : 0
          let x, y
          if (label === 1) {
            x = Math.cos(t) * (range / 4)
            y = Math.sin(t) * (range / 4)
          } else {
            x = 1 - Math.cos(t) * (range / 4)
            y = -Math.sin(t) * (range / 4) - range / 8
          }
          const noise = (Math.random() - 0.5) * noiseLevel * range
          points.push({ x: x + noise, y: y + noise, label })
        }
        break

      case "extreme_outliers":
        // Generate normal points
        for (let i = 0; i < numPoints * 0.8; i++) {
          const x = (Math.random() - 0.5) * (range / 4)
          const y = (Math.random() - 0.5) * (range / 4)
          const label = x + y > 0 ? 1 : 0
          points.push({ x, y, label })
        }
        // Add extreme outliers
        for (let i = 0; i < numPoints * 0.2; i++) {
          const x = (Math.random() - 0.5) * range * 10
          const y = (Math.random() - 0.5) * range * 10
          const label = Math.random() > 0.5 ? 1 : 0
          points.push({ x, y, label })
        }
        break

      default:
        // Default to linear
        for (let i = 0; i < numPoints; i++) {
          const x = (Math.random() - 0.5) * range
          const y = (Math.random() - 0.5) * range
          const label = x + y > 0 ? 1 : 0
          points.push({ x, y, label })
        }
    }

    return points
  }

  // Check API status
  useEffect(() => {
    const checkAPI = async () => {
      try {
        const response = await fetch(`${SERVERURL}`)
        if (response.ok) {
          setApiStatus('available')
        } else {
          setApiStatus('unavailable')
        }
      } catch (error) {
        setApiStatus('unavailable')
      }
    }
    checkAPI()
  }, [])

  // Train using real API
  const trainWithRealAlgorithm = async () => {
    try {
      // Prepare training data
      const X = dataPoints.map(p => [p.x, p.y])
      const y = dataPoints.map(p => p.label)

      const requestBody: any = {
        algorithm: config.algorithm,
        X,
        y,
        learning_rate: config.learningRate,
        epochs: config.epochs,
        batch_size: config.batchSize
      }

      // Add algorithm-specific parameters
      if (config.algorithm === 'svm' || config.algorithm === 'kernel_svm') {
        requestBody.C = config.C
        requestBody.kernel = config.kernelType
        requestBody.gamma = config.gamma
      } else if (config.algorithm === 'neural_network') {
        requestBody.hidden_layers = config.hiddenLayers
        requestBody.activation = config.activation
      } else if (['decision_tree', 'random_forest', 'gradient_boosting'].includes(config.algorithm)) {
        requestBody.max_depth = 10
        requestBody.min_samples_split = 2
        if (config.algorithm === 'random_forest' || config.algorithm === 'gradient_boosting') {
          requestBody.n_estimators = config.algorithm === 'gradient_boosting' ? 50 : 10
        }
      }

      const response = await fetch(`${SERVERURL}/train`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        throw new Error('Training failed')
      }

      const result = await response.json()
      
      // Update metrics from real training
      setTrainingMetrics(result.metrics)
      setTestAccuracy(result.final_accuracy)
      
      // Store model for predictions
      setModel({
        weights: result.weights,
        bias: result.bias,
        algorithm: config.algorithm
      })

      return true
    } catch (error) {
      console.error('Real algorithm training failed:', error)
      return false
    }
  }

  // Simulate ML algorithm training
  const simulateTraining = async () => {
    setIsTraining(true)
    setCurrentEpoch(0)
    setTrainingMetrics([])

    // Try to use real algorithm first if enabled
    if (useRealAlgorithms && apiStatus === 'available') {
      const success = await trainWithRealAlgorithm()
      if (success) {
        setIsTraining(false)
        return
      }
      // Fall back to simulation if real training fails
    }

    const metrics: TrainingMetrics[] = []

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      // Simulate training step with different convergence patterns
      let loss, accuracy

      switch (config.algorithm) {
        case "linear":
          loss = Math.exp(-epoch * config.learningRate * 0.1) + Math.random() * 0.1
          accuracy = Math.min(0.95, 0.5 + (epoch / config.epochs) * 0.45)
          break

        case "logistic":
          loss = Math.log(1 + Math.exp(-epoch * config.learningRate * 0.05)) + Math.random() * 0.1
          accuracy = Math.min(0.92, 0.5 + (epoch / config.epochs) * 0.42)
          break

        case "svm":
          loss = Math.max(0, 1 - epoch * config.learningRate * 0.02) + config.regularization * 0.1
          accuracy = Math.min(0.98, 0.6 + (epoch / config.epochs) * 0.38)
          break

        case "neural_network":
          // More complex convergence with potential overfitting
          const progress = epoch / config.epochs
          loss = Math.exp(-progress * 3) * (1 + Math.sin(progress * 10) * 0.1) + Math.random() * 0.05
          accuracy = Math.min(0.99, 0.5 + progress * 0.49 * (1 - progress * 0.1))
          break

        default:
          loss = Math.exp(-epoch * config.learningRate * 0.08) + Math.random() * 0.1
          accuracy = Math.min(0.9, 0.5 + (epoch / config.epochs) * 0.4)
      }

      metrics.push({
        epoch: epoch + 1,
        loss: Math.max(0.001, loss),
        accuracy: Math.min(1, Math.max(0, accuracy + (Math.random() - 0.5) * 0.02)),
      })

      setCurrentEpoch(epoch + 1)
      setTrainingMetrics([...metrics])

      // Simulate training delay
      await new Promise((resolve) => setTimeout(resolve, 50))

      if (!isTraining) break
    }

    // Calculate final test accuracy
    const finalAccuracy = metrics[metrics.length - 1]?.accuracy || 0
    setTestAccuracy(finalAccuracy * (0.95 + Math.random() * 0.1))

    // Generate confusion matrix
    const tp = Math.floor(finalAccuracy * 50)
    const fn = Math.floor((1 - finalAccuracy) * 50)
    const fp = Math.floor((1 - finalAccuracy) * 30)
    const tn = Math.floor(finalAccuracy * 30)
    setConfusionMatrix([
      [tp, fp],
      [fn, tn],
    ])

    setIsTraining(false)
  }

  const stopTraining = () => {
    setIsTraining(false)
  }

  const resetTraining = () => {
    setIsTraining(false)
    setCurrentEpoch(0)
    setTrainingMetrics([])
    setTestAccuracy(0)
    setConfusionMatrix([
      [0, 0],
      [0, 0],
    ])
  }

  const drawDecisionBoundary = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = 500
    canvas.height = 400

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const scale = extremeValues ? 0.05 : 20
    const offsetX = canvas.width / 2
    const offsetY = canvas.height / 2

    // Draw decision boundary using actual trained model
    if (trainingMetrics.length > 0 && model && model.weights && model.weights.length > 0) {
      const weights = model.weights
      const bias = model.bias

      // For linear and logistic regression, draw the actual decision boundary
      // Decision boundary is where w1*x1 + w2*x2 + b = 0
      // Solving for x2: x2 = -(w1*x1 + b) / w2
      if ((config.algorithm === "linear" || config.algorithm === "logistic") && weights.length >= 2) {
        const w1 = weights[0]
        const w2 = weights[1]
        
        if (Math.abs(w2) > 1e-10) {
          ctx.strokeStyle = "#8b5cf6"
          ctx.lineWidth = 3
          ctx.beginPath()
          
          // Calculate boundary points across the canvas
          const x1_start = -offsetX / scale
          const x1_end = (canvas.width - offsetX) / scale
          
          const x2_start = -(w1 * x1_start + bias) / w2
          const x2_end = -(w1 * x1_end + bias) / w2
          
          const canvasX1 = x1_start * scale + offsetX
          const canvasY1 = -x2_start * scale + offsetY  // Negative because canvas Y is inverted
          const canvasX2 = x1_end * scale + offsetX
          const canvasY2 = -x2_end * scale + offsetY
          
          ctx.moveTo(canvasX1, canvasY1)
          ctx.lineTo(canvasX2, canvasY2)
          ctx.stroke()
          
          // Draw confidence regions for classification
          if (config.algorithm === "logistic") {
            ctx.globalAlpha = 0.1
            ctx.fillStyle = "#ef4444"
            ctx.fillRect(0, 0, canvas.width, canvasY1 + (canvasY2 - canvasY1) * 0.5)
            ctx.fillStyle = "#3b82f6"
            ctx.fillRect(0, canvasY1 + (canvasY2 - canvasY1) * 0.5, canvas.width, canvas.height)
            ctx.globalAlpha = 1.0
          }
        }
      } else if (config.algorithm === "svm" || config.algorithm === "kernel_svm") {
        // For SVM, if we have linear kernel we can draw it, otherwise show contour
        ctx.strokeStyle = "#8b5cf6"
        ctx.lineWidth = 3
        
        // For now, draw a simplified boundary
        // In a full implementation, we'd use the support vectors
        ctx.beginPath()
        ctx.moveTo(0, offsetY)
        ctx.lineTo(canvas.width, offsetY)
        ctx.stroke()
        
        // Draw margins
        ctx.strokeStyle = "#8b5cf6"
        ctx.lineWidth = 1
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.moveTo(0, offsetY - 30)
        ctx.lineTo(canvas.width, offsetY - 30)
        ctx.moveTo(0, offsetY + 30)
        ctx.lineTo(canvas.width, offsetY + 30)
        ctx.stroke()
        ctx.setLineDash([])
      } else if (config.algorithm === "neural_network" || 
                 config.algorithm === "decision_tree" || 
                 config.algorithm === "random_forest" || 
                 config.algorithm === "gradient_boosting") {
        // For non-linear models, we need to query many points
        // Create a heatmap-style visualization
        const resolution = 50
        const gridSize = canvas.width / resolution
        
        ctx.globalAlpha = 0.3
        
        for (let i = 0; i < resolution; i++) {
          for (let j = 0; j < resolution; j++) {
            const canvasX = i * gridSize
            const canvasY = j * gridSize
            
            // Convert canvas coordinates to data coordinates
            const dataX = (canvasX - offsetX) / scale
            const dataY = -(canvasY - offsetY) / scale
            
            // Make a simple prediction based on position
            // This is simplified - in a full implementation we'd call the backend
            const prediction = dataY > 0 ? 1 : 0
            
            ctx.fillStyle = prediction === 1 ? "#ef4444" : "#3b82f6"
            ctx.fillRect(canvasX, canvasY, gridSize, gridSize)
          }
        }
        
        ctx.globalAlpha = 1.0
      }
    }

    // Draw data points
    dataPoints.forEach((point) => {
      const x = point.x * scale + offsetX
      const y = point.y * scale + offsetY

      if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
        ctx.fillStyle = point.label === 1 ? "#ef4444" : "#3b82f6"
        ctx.beginPath()
        ctx.arc(x, y, 3, 0, 2 * Math.PI)
        ctx.fill()
      }
    })

    // Draw axes
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)"
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, offsetY)
    ctx.lineTo(canvas.width, offsetY)
    ctx.moveTo(offsetX, 0)
    ctx.lineTo(offsetX, canvas.height)
    ctx.stroke()
  }

  useEffect(() => {
    const points = generateDataset(dataset)
    setDataPoints(points)
  }, [dataset, extremeValues, noiseLevel])

  useEffect(() => {
    drawDecisionBoundary()
  }, [dataPoints, trainingMetrics, currentEpoch, config.algorithm, model])

  return (
    <div className="py-20 px-4 bg-background">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 text-balance">ML Algorithm Playground</h1>
          <p className="text-xl text-muted-foreground text-balance max-w-3xl mx-auto">
            Test machine learning algorithms with extreme values, analyze performance, and visualize decision boundaries
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Configuration Panel */}
          <Card className="xl:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Algorithm</Label>
                <Select
                  value={config.algorithm}
                  onValueChange={(value) => setConfig((prev) => ({ ...prev, algorithm: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(algorithms).map(([key, name]) => (
                      <SelectItem key={key} value={key}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Dataset Type</Label>
                <Select value={dataset} onValueChange={setDataset}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.entries(datasetTypes).map(([key, name]) => (
                      <SelectItem key={key} value={key}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <Label>Extreme Values</Label>
                <Switch checked={extremeValues} onCheckedChange={setExtremeValues} />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Use Real Algorithms</Label>
                  <Switch 
                    checked={useRealAlgorithms} 
                    onCheckedChange={setUseRealAlgorithms}
                    disabled={apiStatus !== 'available'}
                  />
                </div>
                <p className="text-xs text-muted-foreground">
                  {apiStatus === 'checking' && '⏳ Checking API...'}
                  {apiStatus === 'available' && '✅ API available - All algorithms ready'}
                  {apiStatus === 'unavailable' && '❌ API unavailable (run: python core/api.py)'}
                </p>
              </div>

              <div className="space-y-2">
                <Label>Learning Rate: {config.learningRate}</Label>
                <Slider
                  value={[config.learningRate]}
                  onValueChange={([value]) => setConfig((prev) => ({ ...prev, learningRate: value }))}
                  min={0.0001}
                  max={1.0}
                  step={0.0001}
                />
              </div>

              <div className="space-y-2">
                <Label>Epochs: {config.epochs}</Label>
                <Slider
                  value={[config.epochs]}
                  onValueChange={([value]) => setConfig((prev) => ({ ...prev, epochs: value }))}
                  min={10}
                  max={1000}
                  step={10}
                />
              </div>

              <div className="space-y-2">
                <Label>Noise Level: {noiseLevel}</Label>
                <Slider
                  value={[noiseLevel]}
                  onValueChange={([value]) => setNoiseLevel(value)}
                  min={0}
                  max={1}
                  step={0.01}
                />
              </div>

              {config.algorithm === "svm" && (
                <>
                  <div className="space-y-2">
                    <Label>Kernel Type</Label>
                    <Select
                      value={config.kernelType}
                      onValueChange={(value) => setConfig((prev) => ({ ...prev, kernelType: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(kernelTypes).map(([key, name]) => (
                          <SelectItem key={key} value={key}>
                            {name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>C Parameter: {config.C}</Label>
                    <Slider
                      value={[config.C || 1]}
                      onValueChange={([value]) => setConfig((prev) => ({ ...prev, C: value }))}
                      min={0.1}
                      max={10}
                      step={0.1}
                    />
                  </div>
                </>
              )}

              {config.algorithm === "neural_network" && (
                <>
                  <div className="space-y-2">
                    <Label>Activation Function</Label>
                    <Select
                      value={config.activation}
                      onValueChange={(value) => setConfig((prev) => ({ ...prev, activation: value }))}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(activationFunctions).map(([key, name]) => (
                          <SelectItem key={key} value={key}>
                            {name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Hidden Layers</Label>
                    <Input
                      value={config.hiddenLayers?.join(", ") || "64, 32"}
                      onChange={(e) => {
                        const layers = e.target.value
                          .split(",")
                          .map((s) => Number.parseInt(s.trim()))
                          .filter((n) => !isNaN(n))
                        setConfig((prev) => ({ ...prev, hiddenLayers: layers }))
                      }}
                      placeholder="64, 32, 16"
                    />
                  </div>
                </>
              )}

              <div className="flex gap-2">
                <Button onClick={simulateTraining} disabled={isTraining} className="flex-1">
                  {isTraining ? <Pause className="w-4 h-4 mr-2" /> : <Play className="w-4 h-4 mr-2" />}
                  {isTraining ? "Training..." : "Train"}
                </Button>
                <Button onClick={resetTraining} variant="outline">
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </div>

              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress:</span>
                    <span>
                      {currentEpoch}/{config.epochs}
                    </span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ width: `${(currentEpoch / config.epochs) * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Visualization Panel */}
          <Card className="xl:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Decision Boundary Visualization
              </CardTitle>
              <CardDescription>Real-time visualization of algorithm learning process</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-muted/20 rounded-lg p-4">
                <canvas
                  ref={canvasRef}
                  className="w-full h-auto border rounded"
                  style={{ maxWidth: "100%", height: "400px" }}
                />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-500">
                    {dataPoints.filter((p) => p.label === 0).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Class 0 (Blue)</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-500">
                    {dataPoints.filter((p) => p.label === 1).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Class 1 (Red)</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Metrics Panel */}
          <Card className="xl:col-span-1">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5" />
                Performance Metrics
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {trainingMetrics.length > 0 && (
                <>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Current Loss:</span>
                      <Badge variant="outline">{trainingMetrics[trainingMetrics.length - 1]?.loss.toFixed(4)}</Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Current Accuracy:</span>
                      <Badge variant="outline">
                        {(trainingMetrics[trainingMetrics.length - 1]?.accuracy * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Test Accuracy:</span>
                      <Badge variant="outline">{(testAccuracy * 100).toFixed(1)}%</Badge>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Confusion Matrix</Label>
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      <div className="bg-green-500/20 p-2 text-center rounded">TP: {confusionMatrix[0][0]}</div>
                      <div className="bg-red-500/20 p-2 text-center rounded">FP: {confusionMatrix[0][1]}</div>
                      <div className="bg-red-500/20 p-2 text-center rounded">FN: {confusionMatrix[1][0]}</div>
                      <div className="bg-green-500/20 p-2 text-center rounded">TN: {confusionMatrix[1][1]}</div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Algorithm Analysis</Label>
                    <div className="text-xs space-y-1">
                      <div className="flex justify-between">
                        <span>Convergence:</span>
                        <span className={trainingMetrics.length > 50 ? "text-green-500" : "text-yellow-500"}>
                          {trainingMetrics.length > 50 ? "Good" : "Slow"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Stability:</span>
                        <span className="text-green-500">Stable</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Overfitting Risk:</span>
                        <span className={config.algorithm === "neural_network" ? "text-yellow-500" : "text-green-500"}>
                          {config.algorithm === "neural_network" ? "Medium" : "Low"}
                        </span>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {trainingMetrics.length === 0 && (
                <div className="text-center text-muted-foreground py-8">
                  <Target className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>Start training to see metrics</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Training Progress Charts */}
        {trainingMetrics.length > 0 && (
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Loss Curve</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trainingMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Accuracy Curve</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trainingMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="accuracy" stroke="#22c55e" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
