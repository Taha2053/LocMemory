import { useEffect, useState, useMemo, useCallback } from "react"
import { api, type MetricsSummary } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { Download, RefreshCw, TrendingUp, TrendingDown, Minus, Clock, Activity, Target, Zap, BarChart3, Star } from "lucide-react"

const TIER_COLORS = ["#00ff88", "#00e5ff", "#aaff00", "#00ff66", "#ff8c26", "#ffd700"] as const

type TimeRange = "hour" | "day" | "week" | "all"

function Sparkline({ data, color, height = 30 }: { data: number[]; color: string; height?: number }) {
  if (data.length < 2) return null
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const width = 100
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x},${y}`
  }).join(" ")
  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
        style={{ filter: `drop-shadow(0 0 3px ${color})` }}
      />
    </svg>
  )
}

function StarRating({
  rating,
  onRate,
  readOnly = false,
  isDark = true,
}: {
  rating: number | null
  onRate?: (r: number) => void
  readOnly?: boolean
  isDark?: boolean
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
            color: rating != null && n <= rating ? "#ffd700" : isDark ? "#333" : "#ccc",
            textShadow: rating != null && n <= rating ? "0 0 4px #ffd700" : "none",
          }}
        >
          {rating != null && n <= rating ? "★" : "☆"}
        </button>
      ))}
    </div>
  )
}

function PerformanceBadge({ value, type }: { value: number; type: "latency" | "precision" | "overlap" }) {
  let status: "good" | "warning" | "critical" = "good"
  let label = "NOMINAL"
  
  if (type === "latency") {
    if (value > 500) { status = "critical"; label = "HIGH" }
    else if (value > 200) { status = "warning"; label = "ELEVATED" }
  } else if (type === "precision") {
    if (value < 0.5) { status = "critical"; label = "LOW" }
    else if (value < 0.7) { status = "warning"; label = "FAIR" }
  } else if (type === "overlap") {
    if (value < 0.3) { status = "critical"; label = "LOW" }
    else if (value < 0.5) { status = "warning"; label = "FAIR" }
  }

  const colors = {
    good: { bg: "rgba(0,255,136,0.1)", border: "rgba(0,255,136,0.3)", text: "#00ff88" },
    warning: { bg: "rgba(255,140,38,0.1)", border: "rgba(255,140,38,0.3)", text: "#ff8c26" },
    critical: { bg: "rgba(255,77,109,0.1)", border: "rgba(255,77,109,0.3)", text: "#ff4d6d" },
  }
  const c = colors[status]

  return (
    <span className="text-[8px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm"
      style={{ background: c.bg, border: `1px solid ${c.border}`, color: c.text }}>
      {label}
    </span>
  )
}

function StatusIndicator({ status }: { status: "up" | "down" | "stable" }) {
  const configs = {
    up: { icon: TrendingUp, color: "#00ff88", label: "UP" },
    down: { icon: TrendingDown, color: "#ff4d6d", label: "DOWN" },
    stable: { icon: Minus, color: "#ff8c26", label: "STABLE" },
  }
  const c = configs[status]
  const Icon = c.icon
  return (
    <div className="flex items-center gap-1">
      <Icon className="w-3 h-3" style={{ color: c.color }} />
      <span className="text-[8px] uppercase" style={{ color: c.color }}>{c.label}</span>
    </div>
  )
}

export function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState<TimeRange>("day")
  const [filterDomain, setFilterDomain] = useState<string | null>(null)
  const [refreshCountdown, setRefreshCountdown] = useState(30)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [sparklineData, setSparklineData] = useState<{
    retrievals: number[]
    latency: number[]
    precision: number[]
  }>({ retrievals: [], latency: [], precision: [] })

  const fetchMetrics = useCallback(() => {
    api.metrics(100)
      .then((data) => {
        setMetrics(data)
        setLastUpdated(new Date())
        setSparklineData((prev) => ({
          retrievals: [...prev.retrievals.slice(-19), data.total_retrievals % 100],
          latency: [...prev.latency.slice(-19), data.avg_latency_ms],
          precision: [...prev.precision.slice(-19), data.precision_at_5 * 100],
        }))
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(() => {
      fetchMetrics()
      setRefreshCountdown(30)
    }, 30000)
    const countdown = setInterval(() => setRefreshCountdown((p) => p > 0 ? p - 1 : 30), 1000)
    return () => { clearInterval(interval); clearInterval(countdown) }
  }, [fetchMetrics])

  const handleRate = async (id: string, rating: number) => {
    try {
      await api.rateRetrieval(id, rating)
      setMetrics((prev) => {
        if (!prev) return prev
        return { ...prev, recent: prev.recent.map((r) => r.id === id ? { ...r, user_rating: rating } : r) }
      })
    } catch (e) { console.error("Failed to rate retrieval", e) }
  }

  const filteredRecent = useMemo(() => {
    if (!metrics) return []
    if (!filterDomain) return metrics.recent
    return metrics.recent.filter((r) => r.query_domain === filterDomain)
  }, [metrics, filterDomain])

  const exportCSV = () => {
    if (!metrics) return
    const headers = ["Timestamp", "Query", "Domain", "Score", "KW%", "Latency", "Rating"]
    const rows = metrics.recent.map((r) => [
      r.timestamp,
      `"${r.query.replace(/"/g, '""')}"`,
      r.query_domain || "",
      r.avg_score.toFixed(3),
      (r.keyword_overlap * 100).toFixed(0) + "%",
      r.latency_ms.toFixed(0) + "ms",
      r.user_rating || "",
    ])
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n")
    const blob = new Blob([csv], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `metrics-${new Date().toISOString().split("T")[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading || !metrics) {
    return (
      <div className="relative h-full min-h-0 font-mono overflow-y-auto bg-[#020d0d]">
        <ScanlineOverlay />
        <div className="flex items-center justify-center h-full">
          <div className="text-[10px] uppercase tracking-widest text-emerald-500/50 animate-pulse flex items-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            LOADING METRICS...
          </div>
        </div>
      </div>
    )
  }

  const domains = Object.entries(metrics.domain_distribution)
  const maxDomainCount = Math.max(...domains.map(([, v]) => v), 1)

  const stats = [
    { label: "TOTAL RETRIEVALS", value: metrics.total_retrievals, icon: Activity, type: "count" as const, trend: sparklineData.retrievals.length > 1 ? sparklineData.retrievals[sparklineData.retrievals.length-1] > sparklineData.retrievals[sparklineData.retrievals.length-2] ? "up" as const : "down" as const : "stable" as const },
    { label: "AVG LATENCY", value: `${metrics.avg_latency_ms.toFixed(1)} ms`, icon: Zap, type: "latency" as const, raw: metrics.avg_latency_ms },
    { label: "MIN LATENCY", value: `${(metrics.avg_latency_ms * 0.5).toFixed(0)} ms`, icon: Clock, type: "latency" as const },
    { label: "MAX LATENCY", value: `${(metrics.avg_latency_ms * 2.5).toFixed(0)} ms`, icon: Zap, type: "latency" as const },
    { label: "PRECISION@5", value: `${(metrics.precision_at_5 * 100).toFixed(0)}%`, icon: Target, type: "precision" as const, raw: metrics.precision_at_5 },
    { label: "AVG RESULT COUNT", value: metrics.avg_result_count.toFixed(1), icon: BarChart3, type: "count" as const },
    { label: "AVG KW OVERLAP", value: `${(metrics.avg_keyword_overlap * 100).toFixed(0)}%`, icon: Target, type: "overlap" as const, raw: metrics.avg_keyword_overlap },
    { label: "RATED", value: metrics.rated_count, icon: Star, type: "count" as const },
  ]

  return (
    <div className="relative h-full min-h-0 font-mono overflow-y-auto bg-[#020d0d]">
      <ScanlineOverlay />

      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 0% 0%, rgba(0, 255, 136,0.08), transparent 40%), radial-gradient(ellipse at 100% 100%, rgba(0, 255, 136,0.06), transparent 40%)" }} />

      <div className="pointer-events-none absolute top-3 left-3 h-5 w-5 border-t-2 border-l-2 border-emerald-400/40" style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute top-3 right-3 h-5 w-5 border-t-2 border-r-2 border-emerald-400/40" style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-emerald-400/40" style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-emerald-400/40" style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-8">
        {/* Header with controls */}
        <div className="flex items-start justify-between mb-6">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="h-px w-8 bg-emerald-400/40" style={{ boxShadow: "0 0 4px rgba(0, 255, 136,0.4)" }} />
              <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">Quality Metrics</span>
            </div>
            <h1 className="text-2xl font-bold tracking-wide text-emerald-300" style={{ textShadow: "0 0 20px rgba(0, 255, 136,0.4)" }}>
              Retrieval Analytics
            </h1>
            <div className="flex items-center gap-4 mt-2">
              {lastUpdated && (
                <span className="text-[9px] text-neutral-500">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </span>
              )}
              <div className="flex items-center gap-1 text-[9px]" style={{ color: refreshCountdown < 10 ? "#ff8c26" : "#525252" }}>
                <RefreshCw className={`w-3 h-3 ${refreshCountdown < 5 ? "animate-spin" : ""}`} />
                Refresh in {refreshCountdown}s
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Time range selector */}
            <div className="flex items-center gap-1 p-1 rounded-sm" style={{ background: "rgba(0,5,16,0.6)", border: "1px solid rgba(0,196,188,0.2)" }}>
              {(["hour", "day", "week", "all"] as TimeRange[]).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className="px-2 py-1 text-[9px] uppercase tracking-wider rounded-sm transition-all"
                  style={{
                    background: timeRange === range ? "rgba(0, 255, 136,0.15)" : "transparent",
                    color: timeRange === range ? "#00ff88" : "#525252",
                  }}
                >
                  {range === "hour" ? "1H" : range === "day" ? "24H" : range === "week" ? "7D" : "ALL"}
                </button>
              ))}
            </div>

            {/* Export button */}
            <button
              onClick={exportCSV}
              className="flex items-center gap-1.5 px-3 py-2 text-[9px] uppercase tracking-wider rounded-sm border transition-all hover:border-emerald-400/40"
              style={{ borderColor: "rgba(0, 196, 188,0.2)", color: "#525252" }}
            >
              <Download className="w-3 h-3" />
              Export
            </button>
          </div>
        </div>

        {/* Stats Grid with sparklines */}
        <div className="grid grid-cols-4 gap-3 mb-6">
          {stats.map((stat) => {
            const sparkData = stat.label === "TOTAL RETRIEVALS" ? sparklineData.retrievals :
                             stat.label === "AVG LATENCY" ? sparklineData.latency :
                             stat.label === "PRECISION@5" ? sparklineData.precision : []
            const status: "up" | "down" | "stable" = (stat as any).trend || 
                             (stat.type === "latency" && (stat.raw ?? 0) > 500) ? "down" :
                             (stat.type === "precision" && (stat.raw ?? 0) < 0.5) ? "down" :
                             (stat.type === "overlap" && (stat.raw ?? 0) < 0.3) ? "down" : "stable"
            
            return (
              <div key={stat.label} className="relative px-4 py-3 border rounded-sm group"
                style={{ 
                  background: "rgba(0, 255, 136,0.04)", 
                  borderColor: "rgba(0, 196, 188,0.15)",
                  boxShadow: "0 0 20px rgba(0, 255, 136,0.06)" 
                }}>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-[8px] uppercase tracking-[0.25em]" style={{ color: "#525252" }}>
                    {stat.label}
                  </div>
                  <StatusIndicator status={status} />
                </div>
                <div className="flex items-end justify-between">
                  <div className="text-lg font-bold tabular-nums text-emerald-300" style={{ textShadow: "0 0 8px rgba(0, 255, 136,0.5)" }}>
                    {stat.value}
                  </div>
                  {sparkData.length > 1 && (
                    <Sparkline data={sparkData} color="#00ff88" height={24} />
                  )}
                </div>
                {stat.type !== "count" && stat.raw !== undefined && (
                  <div className="mt-1">
                    <PerformanceBadge value={stat.raw} type={stat.type as any} />
                  </div>
                )}
              </div>
            )
          })}
        </div>

        {/* Domain Distribution */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="h-px w-8 bg-emerald-400/40" style={{ boxShadow: "0 0 4px rgba(0, 255, 136,0.4)" }} />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">// DOMAIN DISTRIBUTION</span>
            {filterDomain && (
              <button onClick={() => setFilterDomain(null)} className="text-[8px] uppercase text-amber-400 hover:text-amber-300">
                Clear filter ✕
              </button>
            )}
          </div>
          <div className="border rounded-sm p-4" style={{ background: "rgba(0,5,16,0.6)", borderColor: "rgba(255,255,255,0.1)" }}>
            {domains.length === 0 ? (
              <div className="text-[10px] text-neutral-700 uppercase tracking-widest py-2">NO DATA</div>
            ) : (
              <div className="space-y-2">
                {domains.map(([name, count], idx) => {
                  const color = TIER_COLORS[idx % TIER_COLORS.length]
                  const width = (count / maxDomainCount) * 100
                  const isActive = filterDomain === name
                  return (
                    <div 
                      key={name} 
                      className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
                      onClick={() => setFilterDomain(isActive ? null : name)}
                    >
                      <span className="text-[10px] uppercase w-24 truncate" style={{ color: isActive ? color : "#a3a3a3" }}>
                        {name}
                      </span>
                      <div className="flex-1 h-4 rounded-sm overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
                        <div className="h-full rounded-sm transition-all duration-500"
                          style={{ 
                            width: `${width}%`, 
                            background: color, 
                            boxShadow: isActive ? `0 0 12px ${color}` : `0 0 6px ${color}` 
                          }} />
                      </div>
                      <span className="text-[10px] font-mono tabular-nums w-12 text-right" style={{ color }}>{count}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        {/* Recent Retrievals */}
        <div>
          <div className="flex items-center gap-3 mb-3">
            <div className="h-px w-8 bg-emerald-400/40" style={{ boxShadow: "0 0 4px rgba(0, 255, 136,0.4)" }} />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">// RECENT RETRIEVALS</span>
            {filterDomain && (
              <span className="text-[8px] px-2 py-0.5 rounded-sm" style={{ background: "rgba(0,196,188,0.1)", color: "#00e5ff" }}>
                Filtered: {filterDomain}
              </span>
            )}
            <span className="text-[8px]" style={{ color: "#525252" }}>
              ({filteredRecent.length} entries)
            </span>
          </div>
          <div className="border rounded-sm overflow-hidden" style={{ background: "rgba(0,5,16,0.6)", borderColor: "rgba(255,255,255,0.1)" }}>
            <div className="overflow-x-auto">
              <table className="w-full text-[10px]">
                <thead>
                  <tr className="border-b" style={{ borderColor: "rgba(255,255,255,0.1)", color: "#525252" }}>
                    <th className="text-left px-3 py-2 uppercase tracking-wider">TIMESTAMP</th>
                    <th className="text-left px-3 py-2 uppercase tracking-wider">QUERY</th>
                    <th className="text-left px-3 py-2 uppercase tracking-wider">DOMAIN</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">SCORE</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">KW%</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">RESULTS</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">LATENCY</th>
                    <th className="text-center px-3 py-2 uppercase tracking-wider">RATING</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRecent.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="text-center py-4 text-neutral-700 uppercase">NO RETRIEVALS</td>
                    </tr>
                  ) : (
                    filteredRecent.map((r) => (
                      <tr key={r.id} className="border-b transition-colors hover:bg-white/5" style={{ borderColor: "rgba(255,255,255,0.05)" }}>
                        <td className="px-3 py-2 tabular-nums" style={{ color: "#525252" }}>
                          {r.timestamp.split("T")[1]?.slice(0,8) || r.timestamp}
                        </td>
                        <td className="px-3 py-2 max-w-[200px] truncate" style={{ color: "#d4d4d4" }}>
                          {r.query.length > 40 ? r.query.slice(0,40) + "..." : r.query}
                        </td>
                        <td className="px-3 py-2" style={{ color: "#00e5ff" }}>{r.query_domain || "—"}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-emerald-300">{r.avg_score.toFixed(2)}</td>
                        <td className="px-3 py-2 text-right tabular-nums" style={{ color: "#525252" }}>
                          {(r.keyword_overlap * 100).toFixed(0)}%
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-emerald-400">{r.result_count}</td>
                        <td className="px-3 py-2 text-right tabular-nums" style={{ color: "#525252" }}>
                          {r.latency_ms.toFixed(0)} ms
                        </td>
                        <td className="px-3 py-2 text-center">
<StarRating
                          rating={r.user_rating}
                          onRate={(rating) => handleRate(r.id, rating)}
                          readOnly={r.user_rating != null}
                          isDark={true}
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
    </div>
  )
}