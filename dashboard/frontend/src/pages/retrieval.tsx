import { useState, useEffect } from "react"
import { api, type RetrieveResponse } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { MatrixRain } from "@/components/MatrixRain"

const TIER_COLORS = ["#00ff88", "#00e5ff", "#aaff00", "#00ff66"] as const

interface RLStatus {
  enabled: boolean
  available: boolean
  model_path?: string
  candidate_pool_size?: number
  top_k?: number
  token_budget?: number
  message?: string
}

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

function ResultCard({ r, idx, rejected = false }: { r: api.RetrievedResult; idx: number; rejected?: boolean }) {
  const color = TIER_COLORS[Math.max(0, r.tier - 1)] ?? TIER_COLORS[0]
  const borderColor = rejected ? "rgba(255,77,109,0.4)" : color

  return (
    <div
      className={`relative rounded-sm overflow-hidden transition-all duration-200 ${rejected ? "opacity-40 cursor-not-allowed" : "hover:scale-[1.003]"}`}
      style={{
        background: "rgba(0,5,16,0.7)",
        border: "1px solid rgba(255,255,255,0.05)",
        borderLeft: `2px solid ${borderColor}`,
      }}
      onMouseEnter={(e) => {
        if (rejected) return
        (e.currentTarget as HTMLElement).style.boxShadow = `0 0 24px ${color}18`
        ;(e.currentTarget as HTMLElement).style.borderColor = `${color}30`
        ;(e.currentTarget as HTMLElement).style.borderLeftColor = color
      }}
      onMouseLeave={(e) => {
        if (rejected) return
        (e.currentTarget as HTMLElement).style.boxShadow = "none"
        ;(e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.05)"
        ;(e.currentTarget as HTMLElement).style.borderLeftColor = color
      }}
    >
      {rejected && (
        <div className="absolute top-3 right-3">
          <span className="text-[9px] font-mono uppercase tracking-wider px-2 py-0.5 rounded-sm"
            style={{ color: "#ff4d6d", background: "rgba(255,77,109,0.15)", border: "1px solid rgba(255,77,109,0.25)" }}>
            REJECTED
          </span>
        </div>
      )}

      <div className="flex gap-4 p-4">
        <div className="text-[22px] font-bold text-neutral-800 shrink-0 tabular-nums leading-none mt-1">
          {String(idx + 1).padStart(2, "0")}
        </div>

        <div className="flex-1 min-w-0 pr-16">
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

          <p className="text-[11px] text-neutral-300 line-clamp-3 leading-relaxed mb-3">{r.text}</p>

          <div className="space-y-1.5">
            <ScoreBar label="COSINE" value={r.cosine_contribution} color="#3b82f6" />
            <ScoreBar label="RECENCY" value={r.recency_contribution} color="#009b94" />
            <ScoreBar label="CATEGORY" value={r.category_contribution} color="#a855f7" />
          </div>
        </div>
      </div>
    </div>
  )
}

export function RetrievalPage() {
  const [query, setQuery] = useState("")
  const [result, setResult] = useState<RetrieveResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [rlStatus, setRlStatus] = useState<RLStatus | null>(null)
  const [includeRejected, setIncludeRejected] = useState(false)
  const [showRejected, setShowRejected] = useState(false)

  useEffect(() => {
    api.rlStatus()
      .then(setRlStatus)
      .catch(() => {})
  }, [])

  const run = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      const res = await api.retrieve(query.trim(), 10, includeRejected)
      setResult(res)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative h-full min-h-0 bg-[#020d0d] font-mono overflow-y-auto">
      <ScanlineOverlay />

      {loading && (
        <MatrixRain
          className="absolute inset-0 h-full w-full pointer-events-none z-[1]"
          opacity={0.08}
          fontSize={12}
        />
      )}

      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 50% 0%, rgba(0, 255, 136,0.08), transparent 50%), radial-gradient(ellipse at 50% 100%, rgba(0, 229, 255,0.06), transparent 50%)" }} />

      <div className="pointer-events-none absolute top-3 left-3 h-5 w-5 border-t-2 border-l-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute top-3 right-3 h-5 w-5 border-t-2 border-r-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 255, 136,0.5))" }} />

      <div className="relative z-10 max-w-2xl mx-auto px-6 py-8">

        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px w-8 bg-emerald-400/40" />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">// SYS.RETRIEVAL</span>
          </div>
          <h1 className="text-2xl font-bold tracking-wide text-emerald-300"
            style={{ textShadow: "0 0 20px rgba(0, 255, 136,0.4)" }}>
            Memory Query Engine
          </h1>
          <div className="mt-2 flex items-center gap-4">
            <p className="text-[11px] text-neutral-500 uppercase tracking-wider">
              Semantic search across the cognitive memory graph
            </p>
            {rlStatus && (
              <span className={`text-[9px] uppercase tracking-wider px-2 py-0.5 rounded-sm ${
                rlStatus.available
                  ? "text-emerald-400 bg-emerald-400/10 border border-emerald-400/20"
                  : rlStatus.enabled
                  ? "text-amber-400 bg-amber-400/10 border border-amber-400/20"
                  : "text-neutral-500 bg-neutral-500/10 border border-neutral-500/20"
              }`}>
                {rlStatus.available ? "RL ACTIVE" : rlStatus.enabled ? "RL LOADING" : "HYBRID MODE"}
              </span>
            )}
          </div>
        </div>

        <div className="relative mb-8">
          <div className="absolute left-0 top-0 bottom-0 w-0.5 transition-all duration-300"
            style={{
              background: loading
                ? "linear-gradient(180deg, transparent, #22d3ee, #a855f7, transparent)"
                : "linear-gradient(180deg, transparent, rgba(0, 255, 136,0.5), transparent)",
              boxShadow: loading ? "0 0 10px rgba(0, 255, 136,0.6)" : "none",
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
            style={{ borderBottom: "1px solid rgba(0, 255, 136,0.25)", caretColor: "#22d3ee" }}
          />
          <div className="absolute right-20 top-1/2 -translate-y-1/2 flex items-center gap-2">
            {rlStatus?.available && (
              <label className="flex items-center gap-1.5 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={includeRejected}
                  onChange={(e) => setIncludeRejected(e.target.checked)}
                  className="sr-only"
                />
                <div className={`w-8 h-4 rounded-full transition-colors ${includeRejected ? "bg-emerald-500/30" : "bg-neutral-800"}`}
                  style={{ border: "1px solid rgba(0, 255, 136,0.3)" }}>
                  <div className={`w-3 h-3 rounded-full mt-0.25 transition-transform ${includeRejected ? "translate-x-4 bg-emerald-400" : "translate-x-0.5 bg-neutral-600"}`}
                    style={{ boxShadow: includeRejected ? "0 0 4px rgba(0, 255, 136,0.6)" : "none" }} />
                </div>
                <span className="text-[9px] uppercase tracking-wider text-neutral-600 group-hover:text-neutral-400 transition-colors">
                  RL
                </span>
              </label>
            )}
          </div>
          <button
            onClick={run}
            disabled={loading || !query.trim()}
            className="absolute right-0 top-1/2 -translate-y-1/2 px-4 py-2 text-[10px] uppercase tracking-widest border transition-all duration-150 disabled:opacity-30"
            style={{
              borderColor: "rgba(0, 255, 136,0.35)",
              color: "rgba(0, 255, 136,0.8)",
              background: loading ? "rgba(0, 255, 136,0.05)" : "transparent",
            }}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="inline-flex gap-1">
                  {[0, 150, 300].map((delay) => (
                    <span key={delay} className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
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

        {result && (
          <div className="space-y-5">
            <div className="flex items-center gap-4 py-2 text-[10px]"
              style={{ borderBottom: "1px solid rgba(0, 255, 136,0.1)" }}>
              <div className="flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full bg-emerald-400"
                  style={{ boxShadow: "0 0 4px rgba(0, 255, 136,0.8)" }} />
                <span className="text-neutral-500 uppercase tracking-wider">
                  DOMAIN: <span className="text-emerald-400">{result.query_domain || "—"}</span>
                </span>
              </div>
              <div className="h-3 w-px bg-white/10" />
              <span className="text-neutral-500 uppercase tracking-wider">
                RESULTS: <span className="text-neutral-300">{result.results.length}</span>
              </span>
              {result.rejected && result.rejected.length > 0 && (
                <>
                  <div className="h-3 w-px bg-white/10" />
                  <span className="text-neutral-500 uppercase tracking-wider">
                    REJECTED: <span className="text-neutral-300">{result.rejected.length}</span>
                  </span>
                </>
              )}
              <div className="ml-auto h-0.5 flex-1 max-w-24"
                style={{ background: "linear-gradient(to right, rgba(0, 255, 136,0.4), transparent)" }} />
            </div>

            <div className="space-y-3">
              {result.results.map((r, idx) => (
                <ResultCard key={r.node_id} r={r} idx={idx} />
              ))}
            </div>

            {result.rejected && result.rejected.length > 0 && (
              <div className="mt-6">
                <button
                  onClick={() => setShowRejected(!showRejected)}
                  className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-neutral-500 hover:text-neutral-300 transition-colors mb-3"
                >
                  <span className={`transition-transform ${showRejected ? "rotate-90" : ""}`}>
                    ▶
                  </span>
                  RL SELECTION — REJECTED CANDIDATES ({result.rejected.length})
                </button>

                {showRejected && (
                  <div className="space-y-3">
                    {result.rejected.map((r, idx) => (
                      <ResultCard key={r.node_id} r={r} idx={idx} rejected />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {!result && !loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-6 relative">
              <div
                className="w-20 h-20 rounded-full border border-emerald-400/15 flex items-center justify-center"
                style={{ boxShadow: "0 0 40px rgba(0, 255, 136,0.08)" }}
              >
                <div className="w-12 h-12 rounded-full border border-emerald-400/20 flex items-center justify-center"
                  style={{ animation: "pulse 3s ease-in-out infinite" }}>
                  <div className="w-2 h-2 rounded-full bg-emerald-500/40"
                    style={{ boxShadow: "0 0 8px rgba(0, 255, 136,0.6)" }} />
                </div>
              </div>
              {[28, 44, 56].map((size, i) => (
                <div key={i}
                  className="absolute top-1/2 left-1/2 rounded-full border border-emerald-400/8"
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