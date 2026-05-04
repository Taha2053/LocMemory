import { useEffect, useState, useMemo, useCallback, useRef } from "react"
import { api, type MetricsSummary } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { domainColor } from "@/lib/domainColors"
import { useTheme } from "@/context/ThemeContext"
import {
  Download, RefreshCw, TrendingUp, TrendingDown, Minus,
  Clock, Activity, Target, Zap, Star,
} from "lucide-react"

type TimeRange = "hour" | "day" | "week" | "all"

const TIME_RANGE_HOURS: Record<TimeRange, number> = {
  hour: 1,
  day: 24,
  week: 168,
  all: Number.POSITIVE_INFINITY,
}

// ─── Stats helpers ────────────────────────────────────────────────────────

type RecentEntry = MetricsSummary["recent"][number]

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length))
  return sorted[idx]
}

function avg(values: number[]): number {
  return values.length ? values.reduce((s, v) => s + v, 0) / values.length : 0
}

function filterByTimeRange(entries: RecentEntry[], range: TimeRange): RecentEntry[] {
  if (range === "all") return entries
  const cutoff = Date.now() - TIME_RANGE_HOURS[range] * 3_600_000
  return entries.filter((e) => {
    const t = new Date(e.timestamp).getTime()
    return Number.isFinite(t) && t >= cutoff
  })
}

interface DomainStats {
  domain: string
  count: number
  avgScore: number
  avgLatency: number
  p95Latency: number
  avgRating: number | null
  ratedCount: number
}

function perDomainStats(entries: RecentEntry[]): DomainStats[] {
  const groups = new Map<string, RecentEntry[]>()
  for (const e of entries) {
    const d = e.query_domain || "unknown"
    const arr = groups.get(d) ?? []
    arr.push(e)
    groups.set(d, arr)
  }
  return Array.from(groups.entries())
    .map(([domain, items]) => {
      const lat = items.map((i) => i.latency_ms)
      const ratings = items.map((i) => i.user_rating).filter((r): r is number => r != null)
      return {
        domain,
        count: items.length,
        avgScore: avg(items.map((i) => i.avg_score)),
        avgLatency: avg(lat),
        p95Latency: percentile(lat, 0.95),
        avgRating: ratings.length ? avg(ratings) : null,
        ratedCount: ratings.length,
      }
    })
    .sort((a, b) => b.count - a.count)
}

function ratingHistogram(entries: RecentEntry[]): { buckets: number[]; unrated: number } {
  const buckets = [0, 0, 0, 0, 0]
  let unrated = 0
  for (const e of entries) {
    if (e.user_rating == null) unrated++
    else buckets[Math.max(0, Math.min(4, e.user_rating - 1))]++
  }
  return { buckets, unrated }
}

function hourlyBuckets(entries: RecentEntry[], hours: number): number[] {
  const now = Date.now()
  const counts = new Array(hours).fill(0)
  for (const e of entries) {
    const t = new Date(e.timestamp).getTime()
    if (!Number.isFinite(t)) continue
    const h = Math.floor((now - t) / 3_600_000)
    if (h >= 0 && h < hours) counts[hours - 1 - h]++
  }
  return counts
}

// ─── UI primitives ────────────────────────────────────────────────────────

function Sparkline({ data, color, height = 28 }: { data: number[]; color: string; height?: number }) {
  if (data.length < 2) return null
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const width = 90
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width
      const y = height - ((v - min) / range) * (height - 4) - 2
      return `${x},${y}`
    })
    .join(" ")
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

function Delta({ pct }: { pct: number | null }) {
  if (pct === null || !Number.isFinite(pct)) {
    return <span className="text-[8px] text-neutral-700 uppercase tracking-wider">—</span>
  }
  const abs = Math.abs(pct)
  if (abs < 0.01) {
    return (
      <div className="flex items-center gap-1 text-[8px] uppercase" style={{ color: "#737373" }}>
        <Minus className="w-2.5 h-2.5" />
        flat
      </div>
    )
  }
  const up = pct > 0
  const Icon = up ? TrendingUp : TrendingDown
  const color = up ? "#00ff88" : "#ff4d6d"
  return (
    <div className="flex items-center gap-1 text-[8px] uppercase tabular-nums" style={{ color }}>
      <Icon className="w-2.5 h-2.5" />
      {(abs * 100).toFixed(1)}%
    </div>
  )
}

function StarRating({
  rating, onRate, readOnly = false,
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

function KpiCard({
  label, value, sub, sparkData, deltaPct, accent,
}: {
  label: string
  value: string
  sub?: string
  sparkData?: number[]
  deltaPct?: number | null
  accent: "ok" | "warn" | "bad"
}) {
  const accentColor = accent === "ok" ? "#00ff88" : accent === "warn" ? "#ff8c26" : "#ff4d6d"
  return (
    <div
      className="relative px-4 py-3 border rounded-sm"
      style={{
        background: "rgba(0,5,16,0.6)",
        borderColor: `${accentColor}25`,
        boxShadow: `0 0 18px ${accentColor}10`,
      }}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[8px] uppercase tracking-[0.25em]" style={{ color: "#737373" }}>
          {label}
        </div>
        {deltaPct !== undefined && <Delta pct={deltaPct ?? null} />}
      </div>
      <div className="flex items-end justify-between gap-2">
        <div
          className="text-lg font-bold tabular-nums"
          style={{ color: accentColor, textShadow: `0 0 8px ${accentColor}50` }}
        >
          {value}
        </div>
        {sparkData && sparkData.length > 1 && <Sparkline data={sparkData} color={accentColor} height={22} />}
      </div>
      {sub && (
        <div className="text-[8px] uppercase tracking-wider mt-1" style={{ color: "#525252" }}>
          {sub}
        </div>
      )}
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 mb-3">
      <div className="h-px w-8 bg-emerald-400/40" style={{ boxShadow: "0 0 4px rgba(0,255,136,0.4)" }} />
      <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/70">{children}</span>
    </div>
  )
}

// ─── Page ────────────────────────────────────────────────────────────────

export function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState<TimeRange>("day")
  const [filterDomain, setFilterDomain] = useState<string | null>(null)
  const [refreshCountdown, setRefreshCountdown] = useState(30)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const { colors } = useTheme()

  // Rolling sparklines for KPI deltas (last ~24 fetches)
  const [series, setSeries] = useState<{
    retrievals: number[]
    latencyP50: number[]
    latencyP95: number[]
    precision: number[]
  }>({ retrievals: [], latencyP50: [], latencyP95: [], precision: [] })

  // Previous-sample snapshot for delta arrows.
  const prevSnap = useRef<{ retrievals: number; p50: number; p95: number; precision: number } | null>(null)

  const fetchMetrics = useCallback(() => {
    api
      .metrics(100)
      .then((data) => {
        setMetrics(data)
        setLastUpdated(new Date())
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    fetchMetrics()
    const interval = setInterval(() => {
      fetchMetrics()
      setRefreshCountdown(30)
    }, 30_000)
    const countdown = setInterval(
      () => setRefreshCountdown((p) => (p > 0 ? p - 1 : 30)),
      1000,
    )
    return () => {
      clearInterval(interval)
      clearInterval(countdown)
    }
  }, [fetchMetrics])

  // ── Derived data, all funneling through timeRange + domain filter ──
  const filteredEntries = useMemo(() => {
    if (!metrics) return []
    let e = filterByTimeRange(metrics.recent, timeRange)
    if (filterDomain) e = e.filter((r) => (r.query_domain || "unknown") === filterDomain)
    return e
  }, [metrics, timeRange, filterDomain])

  // Latencies across the filtered window
  const latencies = useMemo(() => filteredEntries.map((e) => e.latency_ms), [filteredEntries])
  const p50 = useMemo(() => percentile(latencies, 0.5), [latencies])
  const p95 = useMemo(() => percentile(latencies, 0.95), [latencies])
  const precisionWindow = useMemo(() => {
    const rated = filteredEntries.filter((e) => e.user_rating != null)
    if (rated.length === 0) return null
    return rated.filter((e) => (e.user_rating ?? 0) >= 4).length / rated.length
  }, [filteredEntries])

  // Roll forward sparkline buffers + delta snapshot whenever the window changes.
  useEffect(() => {
    if (!metrics) return
    setSeries((prev) => {
      const cap = (arr: number[], v: number) => [...arr.slice(-23), v]
      return {
        retrievals: cap(prev.retrievals, filteredEntries.length),
        latencyP50: cap(prev.latencyP50, p50),
        latencyP95: cap(prev.latencyP95, p95),
        precision: cap(prev.precision, (precisionWindow ?? 0) * 100),
      }
    })
  }, [metrics, filteredEntries.length, p50, p95, precisionWindow])

  // Reset rolling buffers when filter dimensions change so sparklines aren't lying.
  useEffect(() => {
    setSeries({ retrievals: [], latencyP50: [], latencyP95: [], precision: [] })
    prevSnap.current = null
  }, [timeRange, filterDomain])

  const deltas = useMemo(() => {
    if (!metrics) return null
    const cur = {
      retrievals: filteredEntries.length,
      p50,
      p95,
      precision: precisionWindow ?? 0,
    }
    const prev = prevSnap.current
    prevSnap.current = cur
    if (!prev) return { retrievals: null, p50: null, p95: null, precision: null }
    const pct = (a: number, b: number) => (b === 0 ? null : (a - b) / b)
    return {
      retrievals: pct(cur.retrievals, prev.retrievals),
      p50: pct(cur.p50, prev.p50),
      p95: pct(cur.p95, prev.p95),
      precision: pct(cur.precision, prev.precision),
    }
  }, [metrics, filteredEntries.length, p50, p95, precisionWindow])

  const domainStats = useMemo(() => {
    if (!metrics) return []
    // For the domain matrix, ignore the domain filter (we want to see all
    // domains side-by-side) but still respect time range.
    const e = filterByTimeRange(metrics.recent, timeRange)
    return perDomainStats(e)
  }, [metrics, timeRange])

  const ratings = useMemo(() => ratingHistogram(filteredEntries), [filteredEntries])

  const histogramHours =
    timeRange === "hour" ? 12 : timeRange === "day" ? 24 : timeRange === "week" ? 24 * 7 : 24 * 7
  const hourly = useMemo(
    () => hourlyBuckets(filteredEntries, histogramHours),
    [filteredEntries, histogramHours],
  )

  const handleRate = async (id: string, rating: number) => {
    try {
      await api.rateRetrieval(id, rating)
      setMetrics((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          recent: prev.recent.map((r) => (r.id === id ? { ...r, user_rating: rating } : r)),
        }
      })
    } catch (e) {
      console.error("Failed to rate retrieval", e)
    }
  }

  const exportCSV = () => {
    if (!metrics) return
    const headers = ["Timestamp", "Query", "Domain", "Score", "KW%", "Latency", "Rating"]
    const rows = filteredEntries.map((r) => [
      r.timestamp,
      `"${r.query.replace(/"/g, '""')}"`,
      r.query_domain || "",
      r.avg_score.toFixed(3),
      (r.keyword_overlap * 100).toFixed(0) + "%",
      r.latency_ms.toFixed(0) + "ms",
      r.user_rating ?? "",
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
      <div className="relative h-full min-h-0 font-mono overflow-y-auto bg-[#020d0d] custom-scrollbar">
        <ScanlineOverlay />
        <div className="flex items-center justify-center h-full">
          <div className="text-[10px] uppercase tracking-widest animate-pulse flex items-center gap-2" style={{ color: colors.primaryTextDim }}>
            <RefreshCw className="w-4 h-4 animate-spin" />
            LOADING METRICS...
          </div>
        </div>
      </div>
    )
  }

  // Accent thresholds — tuned for a local-first system.
  const p50Accent: "ok" | "warn" | "bad" = p50 < 100 ? "ok" : p50 < 300 ? "warn" : "bad"
  const p95Accent: "ok" | "warn" | "bad" = p95 < 300 ? "ok" : p95 < 800 ? "warn" : "bad"
  const precAccent: "ok" | "warn" | "bad" =
    precisionWindow == null ? "warn" : precisionWindow >= 0.7 ? "ok" : precisionWindow >= 0.5 ? "warn" : "bad"

  return (
    <div className="relative h-full min-h-0 font-mono overflow-y-auto bg-[#020d0d] custom-scrollbar">
      <ScanlineOverlay />

      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: `radial-gradient(ellipse at 0% 0%, ${colors.primaryDim}, transparent 40%), radial-gradient(ellipse at 100% 100%, ${colors.primaryDim}, transparent 40%)`,
        }}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-start justify-between mb-6 gap-4 flex-wrap">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="h-px w-8" style={{ background: colors.primaryBorder }} />
              <span className="text-[9px] uppercase tracking-[0.3em]" style={{ color: colors.primaryTextDim }}>Quality Metrics</span>
            </div>
            <h1
              className="text-2xl font-bold tracking-wide"
              style={{ color: colors.primaryText, textShadow: `0 0 20px ${colors.primaryGlow}` }}
            >
              Retrieval Analytics
            </h1>
            <div className="flex items-center gap-4 mt-2">
              {lastUpdated && (
                <span className="text-[9px] text-neutral-500">
                  Last updated: {lastUpdated.toLocaleTimeString()}
                </span>
              )}
              <div
                className="flex items-center gap-1 text-[9px]"
                style={{ color: refreshCountdown < 10 ? "#ff8c26" : "#525252" }}
              >
                <RefreshCw className={`w-3 h-3 ${refreshCountdown < 5 ? "animate-spin" : ""}`} />
                Refresh in {refreshCountdown}s
              </div>
              <span className="text-[9px] text-neutral-600">
                Window: {filteredEntries.length} entries
                {filterDomain && (
                  <>
                    {" · "}
                    <span style={{ color: domainColor(filterDomain) }}>{filterDomain}</span>
                    <button
                      onClick={() => setFilterDomain(null)}
                      className="ml-1 text-amber-400 hover:text-amber-300"
                    >
                      ✕
                    </button>
                  </>
                )}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div
              className="flex items-center gap-1 p-1 rounded-sm"
              style={{ background: "rgba(0,5,16,0.6)", border: "1px solid rgba(0,196,188,0.2)" }}
            >
              {(["hour", "day", "week", "all"] as TimeRange[]).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className="px-2 py-1 text-[9px] uppercase tracking-wider rounded-sm transition-all"
                  style={{
                    background: timeRange === range ? "rgba(0,255,136,0.15)" : "transparent",
                    color: timeRange === range ? "#00ff88" : "#525252",
                  }}
                >
                  {range === "hour" ? "1H" : range === "day" ? "24H" : range === "week" ? "7D" : "ALL"}
                </button>
              ))}
            </div>

            <button
              onClick={exportCSV}
              className="flex items-center gap-1.5 px-3 py-2 text-[9px] uppercase tracking-wider rounded-sm border transition-all hover:border-emerald-400/40"
              style={{ borderColor: "rgba(0,196,188,0.2)", color: "#525252" }}
            >
              <Download className="w-3 h-3" />
              Export
            </button>
          </div>
        </div>

        {/* KPI ROW — real numbers, real deltas, real sparklines */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <KpiCard
            label="RETRIEVALS"
            value={filteredEntries.length.toLocaleString()}
            sub={`Total all-time: ${metrics.total_retrievals.toLocaleString()}`}
            sparkData={series.retrievals}
            deltaPct={deltas?.retrievals ?? null}
            accent="ok"
          />
          <KpiCard
            label="P50 LATENCY"
            value={`${p50.toFixed(0)} ms`}
            sub={`Median across ${latencies.length} hits`}
            sparkData={series.latencyP50}
            deltaPct={deltas?.p50 ?? null}
            accent={p50Accent}
          />
          <KpiCard
            label="P95 LATENCY"
            value={`${p95.toFixed(0)} ms`}
            sub="Tail latency — what slow queries feel like"
            sparkData={series.latencyP95}
            deltaPct={deltas?.p95 ?? null}
            accent={p95Accent}
          />
          <KpiCard
            label="PRECISION (≥4★)"
            value={precisionWindow == null ? "—" : `${(precisionWindow * 100).toFixed(0)}%`}
            sub={
              precisionWindow == null
                ? "Awaiting ratings"
                : `Of ${filteredEntries.filter((e) => e.user_rating != null).length} rated`
            }
            sparkData={series.precision}
            deltaPct={deltas?.precision ?? null}
            accent={precAccent}
          />
        </div>

        {/* TIMELINE + RATING DISTRIBUTION */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
          <div className="lg:col-span-2">
            <SectionLabel>// RETRIEVALS OVER TIME</SectionLabel>
            <div
              className="border rounded-sm p-4"
              style={{ background: "rgba(0,5,16,0.6)", borderColor: "rgba(255,255,255,0.1)" }}
            >
              {hourly.every((c) => c === 0) ? (
                <div className="text-[10px] text-neutral-700 uppercase tracking-widest py-6 text-center">
                  No retrievals in this window
                </div>
              ) : (
                <>
                  <div className="flex items-end gap-[2px] h-24">
                    {hourly.map((c, i) => {
                      const max = Math.max(...hourly, 1)
                      const h = (c / max) * 100
                      return (
                        <div
                          key={i}
                          className="flex-1 rounded-t-sm transition-all duration-300"
                          style={{
                            height: `${Math.max(h, c > 0 ? 4 : 1)}%`,
                            background:
                              c > 0
                                ? `linear-gradient(180deg, #00ff88 0%, #00cc66 100%)`
                                : "rgba(255,255,255,0.04)",
                            boxShadow: c > 0 ? "0 0 4px rgba(0,255,136,0.5)" : "none",
                          }}
                          title={`${c} retrievals · ${histogramHours - i}h ago`}
                        />
                      )
                    })}
                  </div>
                  <div className="flex justify-between mt-2 text-[8px] text-neutral-700 uppercase tracking-wider">
                    <span>{histogramHours}h ago</span>
                    <span>peak: {Math.max(...hourly)}/h</span>
                    <span>now</span>
                  </div>
                </>
              )}
            </div>
          </div>

          <div>
            <SectionLabel>// USER RATINGS</SectionLabel>
            <div
              className="border rounded-sm p-4 h-full"
              style={{ background: "rgba(0,5,16,0.6)", borderColor: "rgba(255,255,255,0.1)" }}
            >
              {(() => {
                const totalRated = ratings.buckets.reduce((s, v) => s + v, 0)
                const max = Math.max(...ratings.buckets, ratings.unrated, 1)
                if (totalRated === 0 && ratings.unrated === 0) {
                  return (
                    <div className="text-[10px] text-neutral-700 uppercase tracking-widest py-6 text-center">
                      No data
                    </div>
                  )
                }
                const ratingColors = ["#ff4d6d", "#ff8c26", "#ffd700", "#aaff00", "#00ff88"]
                return (
                  <div className="space-y-2">
                    {[5, 4, 3, 2, 1].map((star) => {
                      const c = ratings.buckets[star - 1]
                      const w = (c / max) * 100
                      const color = ratingColors[star - 1]
                      return (
                        <div key={star} className="flex items-center gap-2">
                          <span className="text-[10px] w-8 tabular-nums" style={{ color }}>
                            {star}★
                          </span>
                          <div
                            className="flex-1 h-3 rounded-sm overflow-hidden"
                            style={{ background: "rgba(255,255,255,0.04)" }}
                          >
                            <div
                              className="h-full rounded-sm transition-all duration-500"
                              style={{
                                width: `${w}%`,
                                background: color,
                                boxShadow: c > 0 ? `0 0 6px ${color}80` : "none",
                              }}
                            />
                          </div>
                          <span
                            className="text-[10px] font-mono tabular-nums w-8 text-right"
                            style={{ color }}
                          >
                            {c}
                          </span>
                        </div>
                      )
                    })}
                    <div className="flex items-center gap-2 pt-1 border-t border-white/5 mt-2">
                      <span className="text-[10px] w-8 tabular-nums text-neutral-600">—</span>
                      <div
                        className="flex-1 h-3 rounded-sm overflow-hidden"
                        style={{ background: "rgba(255,255,255,0.04)" }}
                      >
                        <div
                          className="h-full rounded-sm transition-all duration-500"
                          style={{
                            width: `${(ratings.unrated / max) * 100}%`,
                            background: "#525252",
                          }}
                        />
                      </div>
                      <span className="text-[10px] font-mono tabular-nums w-8 text-right text-neutral-600">
                        {ratings.unrated}
                      </span>
                    </div>
                    <div className="text-[8px] uppercase tracking-wider text-neutral-700 pt-1">
                      {totalRated} rated · {ratings.unrated} unrated
                    </div>
                  </div>
                )
              })()}
            </div>
          </div>
        </div>

        {/* PER-DOMAIN QUALITY MATRIX */}
        <div className="mb-6">
          <SectionLabel>// QUALITY BY DOMAIN</SectionLabel>
          <div
            className="border rounded-sm overflow-hidden"
            style={{ background: "rgba(0,5,16,0.6)", borderColor: "rgba(255,255,255,0.1)" }}
          >
            {domainStats.length === 0 ? (
              <div className="text-[10px] text-neutral-700 uppercase tracking-widest py-6 text-center">
                No retrievals in this window
              </div>
            ) : (
              <table className="w-full text-[10px]">
                <thead>
                  <tr
                    className="border-b"
                    style={{ borderColor: "rgba(255,255,255,0.1)", color: "#525252" }}
                  >
                    <th className="text-left px-3 py-2 uppercase tracking-wider">DOMAIN</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">QUERIES</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">AVG SCORE</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">P50 LAT</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">P95 LAT</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">AVG ★</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">RATED</th>
                  </tr>
                </thead>
                <tbody>
                  {domainStats.map((d) => {
                    const c = domainColor(d.domain)
                    const isActive = filterDomain === d.domain
                    return (
                      <tr
                        key={d.domain}
                        className="border-b cursor-pointer transition-colors hover:bg-white/5"
                        style={{
                          borderColor: "rgba(255,255,255,0.05)",
                          background: isActive ? `${c}10` : "transparent",
                        }}
                        onClick={() => setFilterDomain(isActive ? null : d.domain)}
                      >
                        <td className="px-3 py-2 capitalize">
                          <span className="inline-flex items-center gap-2">
                            <span
                              className="inline-block h-2 w-2 rounded-full"
                              style={{ background: c, boxShadow: `0 0 6px ${c}` }}
                            />
                            <span style={{ color: isActive ? c : "#d4d4d4" }}>{d.domain}</span>
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-emerald-400">{d.count}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-emerald-300">
                          {d.avgScore.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums" style={{ color: "#a3a3a3" }}>
                          {d.avgLatency.toFixed(0)} ms
                        </td>
                        <td
                          className="px-3 py-2 text-right tabular-nums"
                          style={{ color: d.p95Latency > 500 ? "#ff8c26" : "#a3a3a3" }}
                        >
                          {d.p95Latency.toFixed(0)} ms
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {d.avgRating == null ? (
                            <span className="text-neutral-700">—</span>
                          ) : (
                            <span style={{ color: "#ffd700" }}>{d.avgRating.toFixed(1)} ★</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-neutral-500">
                          {d.ratedCount}/{d.count}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            )}
          </div>
          <div className="text-[8px] uppercase tracking-wider text-neutral-700 mt-2">
            Click a row to filter the rest of the page · sorted by query count
          </div>
        </div>

        {/* RECENT RETRIEVALS */}
        <div>
          <SectionLabel>// RECENT RETRIEVALS</SectionLabel>
          <div
            className="border rounded-sm overflow-hidden"
            style={{ background: "rgba(0,5,16,0.6)", borderColor: "rgba(255,255,255,0.1)" }}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-[10px]">
                <thead>
                  <tr
                    className="border-b"
                    style={{ borderColor: "rgba(255,255,255,0.1)", color: "#525252" }}
                  >
                    <th className="text-left px-3 py-2 uppercase tracking-wider">TIME</th>
                    <th className="text-left px-3 py-2 uppercase tracking-wider">QUERY</th>
                    <th className="text-left px-3 py-2 uppercase tracking-wider">DOMAIN</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">SCORE</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">KW%</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">N</th>
                    <th className="text-right px-3 py-2 uppercase tracking-wider">LATENCY</th>
                    <th className="text-center px-3 py-2 uppercase tracking-wider">RATING</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEntries.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="text-center py-4 text-neutral-700 uppercase">
                        NO RETRIEVALS
                      </td>
                    </tr>
                  ) : (
                    filteredEntries.slice(0, 50).map((r) => {
                      const c = domainColor(r.query_domain)
                      const slow = r.latency_ms > 500
                      return (
                        <tr
                          key={r.id}
                          className="border-b transition-colors hover:bg-white/5"
                          style={{ borderColor: "rgba(255,255,255,0.05)" }}
                        >
                          <td className="px-3 py-2 tabular-nums" style={{ color: "#525252" }}>
                            {r.timestamp.split("T")[1]?.slice(0, 8) || r.timestamp}
                          </td>
                          <td className="px-3 py-2 max-w-[280px] truncate" style={{ color: "#d4d4d4" }}>
                            {r.query.length > 50 ? r.query.slice(0, 50) + "…" : r.query}
                          </td>
                          <td className="px-3 py-2">
                            {r.query_domain ? (
                              <span className="inline-flex items-center gap-1.5">
                                <span
                                  className="inline-block h-1.5 w-1.5 rounded-full"
                                  style={{ background: c, boxShadow: `0 0 4px ${c}` }}
                                />
                                <span style={{ color: c }}>{r.query_domain}</span>
                              </span>
                            ) : (
                              <span className="text-neutral-700">—</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-emerald-300">
                            {r.avg_score.toFixed(2)}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums" style={{ color: "#525252" }}>
                            {(r.keyword_overlap * 100).toFixed(0)}%
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-emerald-400">
                            {r.result_count}
                          </td>
                          <td
                            className="px-3 py-2 text-right tabular-nums"
                            style={{ color: slow ? "#ff8c26" : "#525252" }}
                          >
                            {r.latency_ms.toFixed(0)} ms
                          </td>
                          <td className="px-3 py-2 text-center">
                            <StarRating
                              rating={r.user_rating}
                              onRate={(rating) => handleRate(r.id, rating)}
                              readOnly={r.user_rating != null}
                            />
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
          {filteredEntries.length > 50 && (
            <div className="text-[8px] uppercase tracking-wider text-neutral-700 mt-2">
              Showing first 50 of {filteredEntries.length} · use Export for full data
            </div>
          )}
        </div>

        {/* FOOTER LEGEND */}
        <div className="mt-6 flex items-center gap-4 text-[8px] uppercase tracking-wider text-neutral-700">
          <span className="flex items-center gap-1">
            <Activity className="w-2.5 h-2.5" /> volume
          </span>
          <span className="flex items-center gap-1">
            <Zap className="w-2.5 h-2.5" /> latency
          </span>
          <span className="flex items-center gap-1">
            <Target className="w-2.5 h-2.5" /> precision
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-2.5 h-2.5" /> recency
          </span>
          <span className="flex items-center gap-1">
            <Star className="w-2.5 h-2.5" /> user rating
          </span>
        </div>
      </div>
    </div>
  )
}
