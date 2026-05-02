import { useEffect, useState } from "react"
import { api, type MetricsSummary } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"

const TIER_COLORS = ["#00ff88", "#00e5ff", "#aaff00", "#00ff66", "#ff8c26", "#ffd700"] as const

function StarRating({
  rating,
  onRate,
  readOnly = false,
}: {
  rating: number | null
  onRate?: (r: number) => void
  readOnly?: boolean
}) {
  return (
    <div className="flex gap-0.5">
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          disabled={readOnly || !onRate}
          onClick={() => !readOnly && onRate?.(n)}
          className="text-sm leading-none transition-colors"
          style={{
            color: rating != null && n <= rating ? "#ffd700" : "#333",
            textShadow: rating != null && n <= rating ? "0 0 4px #ffd700" : "none",
          }}
        >
          {rating != null && n <= rating ? "★" : "☆"}
        </button>
      ))}
    </div>
  )
}

export function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchMetrics = () => {
    api.metrics(100)
      .then(setMetrics)
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    fetchMetrics()
    const id = setInterval(fetchMetrics, 30000)
    return () => clearInterval(id)
  }, [])

  const handleRate = async (id: string, rating: number) => {
    try {
      await api.rateRetrieval(id, rating)
      setMetrics((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          recent: prev.recent.map((r) =>
            r.id === id ? { ...r, user_rating: rating } : r
          ),
        }
      })
    } catch (e) {
      console.error("Failed to rate retrieval", e)
    }
  }

  if (loading || !metrics) {
    return (
      <div className="relative h-full min-h-0 bg-[#020d0d] font-mono overflow-y-auto">
        <ScanlineOverlay />
        <div className="flex items-center justify-center h-full">
          <div className="text-[10px] uppercase tracking-widest text-emerald-500/50 animate-pulse">
            LOADING METRICS...
          </div>
        </div>
      </div>
    )
  }

  const domains = Object.entries(metrics.domain_distribution)
  const maxDomainCount = Math.max(...domains.map(([, v]) => v), 1)

  return (
    <div className="relative h-full min-h-0 bg-[#020d0d] font-mono overflow-y-auto">
      <ScanlineOverlay />

      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 0% 0%, rgba(0, 255, 136,0.08), transparent 40%), radial-gradient(ellipse at 100% 100%, rgba(0, 255, 136,0.06), transparent 40%)" }} />

      <div className="pointer-events-none absolute top-3 left-3 h-5 w-5 border-t-2 border-l-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute top-3 right-3 h-5 w-5 border-t-2 border-r-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-8">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px w-8 bg-emerald-400/40" style={{ boxShadow: "0 0 4px rgba(0, 255, 136,0.4)" }} />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">// SYS.METRICS</span>
          </div>
          <h1 className="text-2xl font-bold tracking-wide text-emerald-300"
            style={{ textShadow: "0 0 20px rgba(0, 255, 136,0.4)" }}>
            Retrieval Analytics
          </h1>
          <p className="mt-1 text-[11px] text-neutral-500 uppercase tracking-wider">
            Live performance metrics — auto-refresh every 30s
          </p>
        </div>

        <div className="grid grid-cols-4 gap-3 mb-8">
          {[
            { label: "TOTAL RETRIEVALS", value: metrics.total_retrievals },
            { label: "AVG LATENCY", value: `${metrics.avg_latency_ms.toFixed(1)} ms` },
            { label: "PRECISION@5", value: `${(metrics.precision_at_5 * 100).toFixed(0)}%` },
            { label: "AVG KW OVERLAP", value: `${(metrics.avg_keyword_overlap * 100).toFixed(0)}%` },
          ].map(({ label, value }) => (
            <div key={label}
              className="px-4 py-3 border border-emerald-400/15 rounded-sm"
              style={{ background: "rgba(0, 255, 136,0.04)", boxShadow: "0 0 20px rgba(0, 255, 136,0.06)" }}>
              <div className="text-[8px] uppercase tracking-[0.25em] text-neutral-600 mb-1">{label}</div>
              <div className="text-lg font-bold tabular-nums text-emerald-300"
                style={{ textShadow: "0 0 8px rgba(0, 255, 136,0.5)" }}>
                {typeof value === "number" ? value.toLocaleString() : value}
              </div>
            </div>
          ))}
        </div>

        <div className="mb-8">
          <div className="flex items-center gap-3 mb-3">
            <div className="h-px w-8 bg-emerald-400/40" style={{ boxShadow: "0 0 4px rgba(0, 255, 136,0.4)" }} />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">// DOMAIN DISTRIBUTION</span>
          </div>
          <div className="border border-white/5 rounded-sm p-4" style={{ background: "rgba(0,5,16,0.6)" }}>
            {domains.length === 0 ? (
              <div className="text-[10px] text-neutral-700 uppercase tracking-widest py-2">NO DATA</div>
            ) : (
              <div className="space-y-2">
                {domains.map(([name, count], idx) => {
                  const color = TIER_COLORS[idx % TIER_COLORS.length]
                  const width = (count / maxDomainCount) * 100
                  return (
                    <div key={name} className="flex items-center gap-3">
                      <span className="text-[10px] uppercase w-24 truncate" style={{ color: "#a3a3a3" }}>{name}</span>
                      <div className="flex-1 h-4 bg-white/5 rounded-sm overflow-hidden">
                        <div className="h-full rounded-sm transition-all duration-500"
                          style={{ width: `${width}%`, background: color, boxShadow: `0 0 6px ${color}` }} />
                      </div>
                      <span className="text-[10px] font-mono tabular-nums w-12 text-right" style={{ color }}>{count}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        <div>
          <div className="flex items-center gap-3 mb-3">
            <div className="h-px w-8 bg-emerald-400/40" style={{ boxShadow: "0 0 4px rgba(0, 255, 136,0.4)" }} />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">// RECENT RETRIEVALS</span>
          </div>
          <div className="border border-white/5 rounded-sm overflow-hidden" style={{ background: "rgba(0,5,16,0.6)" }}>
            <table className="w-full text-[10px]">
              <thead>
                <tr className="border-b border-white/5 text-neutral-600 uppercase tracking-wider">
                  <th className="text-left px-3 py-2">TIMESTAMP</th>
                  <th className="text-left px-3 py-2">QUERY</th>
                  <th className="text-left px-3 py-2">DOMAIN</th>
                  <th className="text-right px-3 py-2">SCORE</th>
                  <th className="text-right px-3 py-2">KW%</th>
                  <th className="text-right px-3 py-2">LATENCY</th>
                  <th className="text-center px-3 py-2">RATING</th>
                </tr>
              </thead>
              <tbody>
                {metrics.recent.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="text-center py-4 text-neutral-700 uppercase">NO RETRIEVALS</td>
                  </tr>
                ) : (
                  metrics.recent.map((r) => (
                    <tr key={r.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                      <td className="px-3 py-2 tabular-nums text-neutral-500">{r.timestamp.split("T")[1]?.slice(0,8) || r.timestamp}</td>
                      <td className="px-3 py-2 text-neutral-300 max-w-[200px] truncate">{r.query.length > 40 ? r.query.slice(0,40) + "..." : r.query}</td>
                      <td className="px-3 py-2 text-emerald-400/80">{r.query_domain || "—"}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-emerald-300">{r.avg_score.toFixed(2)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-neutral-400">{(r.keyword_overlap * 100).toFixed(0)}%</td>
                      <td className="px-3 py-2 text-right tabular-nums text-neutral-400">{r.latency_ms.toFixed(0)} ms</td>
                      <td className="px-3 py-2 text-center">
                        <StarRating
                          rating={r.user_rating}
                          onRate={(rating) => handleRate(r.id, rating)}
                          readOnly={r.user_rating != null}
                        />
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}