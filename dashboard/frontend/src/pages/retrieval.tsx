import { useState } from "react"
import { api, type RetrieveResponse } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { MatrixRain } from "@/components/MatrixRain"

const TIER_COLORS = ["#3b82f6", "#06b6d4", "#9ec5e8", "#a855f7"] as const

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.max(0, Math.min(100, value * 100))
  return (
    <div className="flex items-center gap-2 text-[9px]">
      <span className="text-neutral-600 w-14 uppercase tracking-wider">{label}</span>
      <div className="flex-1 h-0.5 bg-white/5 overflow-hidden rounded-full">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ width: `${pct}%`, background: color, boxShadow: `0 0 4px ${color}` }}
        />
      </div>
      <span className="text-neutral-500 w-8 text-right font-mono tabular-nums">{value.toFixed(2)}</span>
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
      const res = await api.retrieve(query.trim())
      setResult(res)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative h-full min-h-0 bg-[#000510] font-mono overflow-y-auto">
      <ScanlineOverlay />

      {/* Matrix rain on loading */}
      {loading && (
        <MatrixRain
          className="absolute inset-0 h-full w-full pointer-events-none z-[1]"
          opacity={0.08}
          fontSize={12}
        />
      )}

      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 50% 0%, rgba(59,130,246,0.08), transparent 50%), radial-gradient(ellipse at 50% 100%, rgba(168,85,247,0.06), transparent 50%)" }} />

      {/* Corner brackets */}
      <div className="pointer-events-none absolute top-3 left-3 h-5 w-5 border-t-2 border-l-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />
      <div className="pointer-events-none absolute top-3 right-3 h-5 w-5 border-t-2 border-r-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />

      <div className="relative z-10 max-w-2xl mx-auto px-6 py-8">

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px w-8 bg-cyan-400/40" />
            <span className="text-[9px] uppercase tracking-[0.3em] text-cyan-600/60">// SYS.RETRIEVAL</span>
          </div>
          <h1 className="text-2xl font-bold tracking-wide text-cyan-300"
            style={{ textShadow: "0 0 20px rgba(34,211,238,0.4)" }}>
            Memory Query Engine
          </h1>
          <p className="mt-1 text-[11px] text-neutral-500 uppercase tracking-wider">
            Semantic search across the cognitive memory graph
          </p>
        </div>

        {/* Query input */}
        <div className="relative mb-8">
          <div className="absolute left-0 top-0 bottom-0 w-0.5 transition-all duration-300"
            style={{
              background: loading
                ? "linear-gradient(180deg, transparent, #22d3ee, #a855f7, transparent)"
                : "linear-gradient(180deg, transparent, rgba(6,182,212,0.5), transparent)",
              boxShadow: loading ? "0 0 10px rgba(34,211,238,0.6)" : "none",
              animation: loading ? "pulse 1s ease-in-out infinite" : "none",
            }} />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="QUERY THE MEMORY GRAPH..."
            disabled={loading}
            className="w-full bg-transparent py-4 pl-4 pr-28 text-base text-neutral-100 placeholder:text-neutral-700 focus:outline-none transition-all"
            style={{ borderBottom: "1px solid rgba(6,182,212,0.25)", caretColor: "#22d3ee" }}
          />
          <button
            onClick={run}
            disabled={loading || !query.trim()}
            className="absolute right-0 top-1/2 -translate-y-1/2 px-4 py-2 text-[10px] uppercase tracking-widest border transition-all duration-150 disabled:opacity-30"
            style={{
              borderColor: "rgba(34,211,238,0.35)",
              color: "rgba(34,211,238,0.8)",
              background: loading ? "rgba(34,211,238,0.05)" : "transparent",
            }}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="inline-flex gap-1">
                  {[0, 150, 300].map((delay) => (
                    <span key={delay} className="w-1 h-1 bg-cyan-400 rounded-full animate-bounce"
                      style={{ animationDelay: `${delay}ms` }} />
                  ))}
                </span>
                SCANNING
              </span>
            ) : (
              "[ SCAN ]"
            )}
          </button>
        </div>

        {/* Results */}
        {result && (
          <div className="space-y-5">
            {/* Result meta bar */}
            <div className="flex items-center gap-4 py-2 text-[10px]"
              style={{ borderBottom: "1px solid rgba(6,182,212,0.1)" }}>
              <div className="flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full bg-cyan-400"
                  style={{ boxShadow: "0 0 4px rgba(34,211,238,0.8)" }} />
                <span className="text-neutral-500 uppercase tracking-wider">
                  DOMAIN: <span className="text-cyan-400">{result.query_domain || "—"}</span>
                </span>
              </div>
              <div className="h-3 w-px bg-white/10" />
              <span className="text-neutral-500 uppercase tracking-wider">
                RESULTS: <span className="text-neutral-300">{result.results.length}</span>
              </span>
              <div className="ml-auto h-0.5 flex-1 max-w-24"
                style={{ background: "linear-gradient(to right, rgba(34,211,238,0.4), transparent)" }} />
            </div>

            {/* Result cards */}
            <div className="space-y-3">
              {result.results.map((r, idx) => {
                const color = TIER_COLORS[r.tier] || TIER_COLORS[0]
                return (
                  <div
                    key={r.node_id}
                    className="relative rounded-sm overflow-hidden transition-all duration-200 hover:scale-[1.003]"
                    style={{
                      background: "rgba(0,5,16,0.7)",
                      border: "1px solid rgba(255,255,255,0.05)",
                      borderLeft: `2px solid ${color}`,
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.boxShadow = `0 0 24px ${color}18`
                      ;(e.currentTarget as HTMLElement).style.borderColor = `${color}30`
                      ;(e.currentTarget as HTMLElement).style.borderLeftColor = color
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.boxShadow = "none"
                      ;(e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.05)"
                      ;(e.currentTarget as HTMLElement).style.borderLeftColor = color
                    }}
                  >
                    {/* Score badge */}
                    <div className="absolute top-3 right-3">
                      <span className="text-[10px] font-mono tabular-nums px-2 py-0.5 rounded-sm"
                        style={{ color, background: `${color}18`, border: `1px solid ${color}25` }}>
                        {r.score.toFixed(3)}
                      </span>
                    </div>

                    <div className="flex gap-4 p-4">
                      {/* Index number */}
                      <div className="text-[22px] font-bold text-neutral-800 shrink-0 tabular-nums leading-none mt-1">
                        {String(idx + 1).padStart(2, "0")}
                      </div>

                      <div className="flex-1 min-w-0 pr-16">
                        {/* Domain breadcrumb */}
                        <div className="flex flex-wrap items-center gap-1.5 mb-2">
                          <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm"
                            style={{ color, background: `${color}15` }}>
                            {r.domain || "—"}
                          </span>
                          {r.subdomain && (
                            <>
                              <span className="text-neutral-700 text-[9px]">›</span>
                              <span className="text-[9px] text-neutral-600">{r.subdomain}</span>
                            </>
                          )}
                        </div>

                        {/* Text */}
                        <p className="text-[11px] text-neutral-300 line-clamp-3 leading-relaxed mb-3">{r.text}</p>

                        {/* Score bars */}
                        <div className="space-y-1.5">
                          <ScoreBar label="COSINE"   value={r.cosine_contribution}   color="#3b82f6" />
                          <ScoreBar label="RECENCY"  value={r.recency_contribution}  color="#06b6d4" />
                          <ScoreBar label="CATEGORY" value={r.category_contribution} color="#a855f7" />
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!result && !loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-6 relative">
              <div
                className="w-20 h-20 rounded-full border border-cyan-400/15 flex items-center justify-center"
                style={{ boxShadow: "0 0 40px rgba(6,182,212,0.08)" }}
              >
                <div className="w-12 h-12 rounded-full border border-cyan-400/20 flex items-center justify-center"
                  style={{ animation: "pulse 3s ease-in-out infinite" }}>
                  <div className="w-2 h-2 rounded-full bg-cyan-500/40"
                    style={{ boxShadow: "0 0 8px rgba(34,211,238,0.6)" }} />
                </div>
              </div>
              {/* Orbit rings */}
              {[28, 44, 56].map((size, i) => (
                <div key={i}
                  className="absolute top-1/2 left-1/2 rounded-full border border-cyan-400/8"
                  style={{
                    width: size * 2,
                    height: size * 2,
                    marginLeft: -size,
                    marginTop: -size,
                    animation: `spin ${6 + i * 2}s linear infinite`,
                    animationDirection: i % 2 === 0 ? "normal" : "reverse",
                  }} />
              ))}
            </div>
            <div className="text-[9px] uppercase tracking-[0.3em] text-neutral-700 mb-1">// STANDBY</div>
            <div className="text-[11px] text-neutral-600 uppercase tracking-widest animate-pulse">
              AWAITING QUERY INPUT
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
