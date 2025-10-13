"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts"
import { Clock, Zap, Target, TrendingUp } from "lucide-react"

const convergenceData = [
  { iteration: 0, GD: 100, SGD: 100, Adam: 100, MiniBatch: 100 },
  { iteration: 10, GD: 85, SGD: 75, Adam: 60, MiniBatch: 80 },
  { iteration: 20, GD: 72, SGD: 55, Adam: 35, MiniBatch: 65 },
  { iteration: 30, GD: 61, SGD: 42, Adam: 20, MiniBatch: 52 },
  { iteration: 40, GD: 52, SGD: 35, Adam: 12, MiniBatch: 42 },
  { iteration: 50, GD: 45, SGD: 30, Adam: 8, MiniBatch: 35 },
]

const complexityData = [
  { algorithm: "GD", timeComplexity: "O(n)", spaceComplexity: "O(n)", convergenceRate: "Linear" },
  { algorithm: "SGD", timeComplexity: "O(1)", spaceComplexity: "O(1)", convergenceRate: "Sub-linear" },
  { algorithm: "Mini-Batch", timeComplexity: "O(k)", spaceComplexity: "O(k)", convergenceRate: "Linear" },
  { algorithm: "Adam", timeComplexity: "O(n)", spaceComplexity: "O(n)", convergenceRate: "Fast" },
  { algorithm: "Analytical", timeComplexity: "O(n³)", spaceComplexity: "O(n²)", convergenceRate: "Instant" },
  { algorithm: "QP", timeComplexity: "O(n³)", spaceComplexity: "O(n²)", convergenceRate: "Polynomial" },
]

const performanceData = [
  { metric: "Speed", GD: 60, SGD: 95, Adam: 85, MiniBatch: 75 },
  { metric: "Stability", GD: 90, SGD: 40, Adam: 80, MiniBatch: 85 },
  { metric: "Memory", GD: 30, SGD: 95, Adam: 30, MiniBatch: 70 },
  { metric: "Accuracy", GD: 85, SGD: 70, Adam: 90, MiniBatch: 80 },
]

export function AnalysisSection() {
  return (
    <section id="analysis" className="py-20 px-4 bg-muted/20">
      <div className="container mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold mb-4 text-balance">Algorithm Analysis & Comparison</h2>
          <p className="text-xl text-muted-foreground text-balance max-w-2xl mx-auto">
            Deep dive into the computational complexity and performance characteristics of each optimizer
          </p>
        </div>

        <Tabs defaultValue="convergence" className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
            <TabsTrigger value="convergence">Convergence</TabsTrigger>
            <TabsTrigger value="complexity">Complexity</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="daa">DAA Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="convergence" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5" />
                  Convergence Comparison
                </CardTitle>
                <CardDescription>Loss reduction over iterations for different optimization algorithms</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={convergenceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="iteration" stroke="hsl(var(--muted-foreground))" />
                      <YAxis stroke="hsl(var(--muted-foreground))" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Line type="monotone" dataKey="GD" stroke="#3b82f6" strokeWidth={2} />
                      <Line type="monotone" dataKey="SGD" stroke="#10b981" strokeWidth={2} />
                      <Line type="monotone" dataKey="Adam" stroke="#f97316" strokeWidth={2} />
                      <Line type="monotone" dataKey="MiniBatch" stroke="#8b5cf6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="complexity" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {complexityData.map((item) => (
                <Card key={item.algorithm}>
                  <CardHeader>
                    <CardTitle className="text-lg">{item.algorithm}</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Time Complexity:</span>
                      <Badge variant="outline" className="font-mono">
                        {item.timeComplexity}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Space Complexity:</span>
                      <Badge variant="outline" className="font-mono">
                        {item.spaceComplexity}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Convergence:</span>
                      <Badge variant="secondary">{item.convergenceRate}</Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Performance Metrics
                </CardTitle>
                <CardDescription>Comparative analysis across key performance dimensions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="metric" stroke="hsl(var(--muted-foreground))" />
                      <YAxis stroke="hsl(var(--muted-foreground))" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="GD" fill="#3b82f6" />
                      <Bar dataKey="SGD" fill="#10b981" />
                      <Bar dataKey="Adam" fill="#f97316" />
                      <Bar dataKey="MiniBatch" fill="#8b5cf6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="daa" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Time Complexity Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-2">Gradient Descent</h4>
                      <p className="text-xs text-muted-foreground">
                        Linear in dataset size. Each iteration processes all n samples.
                      </p>
                    </div>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-2">SGD</h4>
                      <p className="text-xs text-muted-foreground">
                        Constant time per iteration. Processes one sample at a time.
                      </p>
                    </div>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-2">Analytical</h4>
                      <p className="text-xs text-muted-foreground">
                        Cubic complexity due to matrix inversion operations.
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5" />
                    Convergence Theory
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-2">Linear Convergence</h4>
                      <p className="text-xs text-muted-foreground">
                        {String.raw`||x_k - x^*|| ≤ c^k ||x_0 - x^*|| where 0 < c < 1`}
                      </p>
                    </div>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-2">Sub-linear Convergence</h4>
                      <p className="text-xs text-muted-foreground">{String.raw`||x_k - x^*|| ≤ C/k^α where α > 0`}</p>
                    </div>
                    <div className="p-3 bg-muted/50 rounded-lg">
                      <h4 className="font-semibold text-sm mb-2">Quadratic Convergence</h4>
                      <p className="text-xs text-muted-foreground">
                        {String.raw`||x_{k+1} - x^*|| ≤ C ||x_k - x^*||^2`}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  )
}
