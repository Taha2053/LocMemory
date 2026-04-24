import { useState } from "react"
import { api, type RetrieveResponse } from "@/lib/api"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

function Bar({
  label,
  value,
  color,
}: {
  label: string
  value: number
  color: string
}) {
  const pct = Math.max(0, Math.min(100, value * 100))
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
        <div className="h-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

export function RetrievalPage() {
  const [query, setQuery] = useState("")
  const [result, setResult] = useState<RetrieveResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const run = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      setResult(await api.retrieve(query.trim()))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 space-y-4 max-w-4xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Retrieval</h1>
        <p className="text-sm text-muted-foreground">
          Type a query to see which memories are retrieved and why (cosine / recency / category breakdown).
        </p>
      </header>

      <div className="flex gap-2">
        <Input
          placeholder="e.g. what programming languages do I know?"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        <Button onClick={run} disabled={loading}>
          {loading ? "…" : "Retrieve"}
        </Button>
      </div>

      {result && (
        <div className="space-y-3">
          <div className="text-sm text-muted-foreground">
            query domain: <Badge variant="outline">{result.query_domain}</Badge> · {result.results.length} results
          </div>
          {result.results.map((r) => (
            <Card key={r.node_id}>
              <CardContent className="p-4 grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4">
                <div className="space-y-2 min-w-0">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <Badge variant="secondary">{r.domain || "(none)"}</Badge>
                    {r.subdomain && <Badge variant="outline">{r.subdomain}</Badge>}
                    <span className="text-xs text-muted-foreground">
                      score {r.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="text-sm">{r.text}</p>
                </div>
                <div className="space-y-2">
                  <Bar label="cosine" value={r.cosine_contribution} color="#3b82f6" />
                  <Bar label="recency" value={r.recency_contribution} color="#10b981" />
                  <Bar label="category" value={r.category_contribution} color="#f59e0b" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
