import { useState, useEffect, useRef, useCallback } from "react"
import { api, type RetrievedResult, type RetrieveResponse } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { useTheme } from "@/context/ThemeContext"

// ─── Constants ────────────────────────────────────────────────────────────
const TIER_COLORS: Record<number, string> = {
  1: "#00ff88",
  2: "#00e5ff",
  3: "#aaff00",
  4: "#ff8c26",
}
function tierColor(t: number) { return TIER_COLORS[t] ?? "#00ff88" }

const SCORE_META = [
  { key: "cosine_contribution",   label: "Semantic",  desc: "Vector similarity — how close the meaning is",   color: "#60a5fa" },
  { key: "recency_contribution",  label: "Recency",   desc: "Prefers newer memories over older ones",         color: "#34d399" },
  { key: "category_contribution", label: "Domain",    desc: "Domain/category match with the query",          color: "#a78bfa" },
] as const

// ─── Pipeline step definitions ─────────────────────────────────────────────
const STEPS = [
  { id: "embed",     label: "Embed query",       desc: "Query → semantic vector (embedding model)" },
  { id: "classify",  label: "Classify domain",   desc: "Detect the knowledge domain of the query" },
  { id: "pool",      label: "Build candidate pool", desc: "Hybrid score: semantic + recency + domain match" },
  { id: "rl",        label: "RL reranking",       desc: "Trained agent selects best k from the pool" },
  { id: "return",    label: "Return results",     desc: "Final ranked list delivered to you" },
]

// ─── RLStatus type ─────────────────────────────────────────────────────────
interface RLStatus {
  enabled: boolean
  available: boolean
  model_path?: string
  candidate_pool_size?: number
  top_k?: number
  token_budget?: number
  message?: string
}

// ─── Animated pipeline ─────────────────────────────────────────────────────
function PipelineSteps({
  active,
  rlAvailable,
}: {
  active: boolean
  rlAvailable: boolean
}) {
  const [current, setCurrent] = useState(-1)

  useEffect(() => {
    if (!active) { setCurrent(-1); return }
    setCurrent(0)
    const timings = [300, 700, 1200, 2000]
    const timers = timings.map((ms, i) =>
      setTimeout(() => setCurrent(i + 1), ms)
    )
    return () => timers.forEach(clearTimeout)
  }, [active])

  return (
    <div className="space-y-2">
      {STEPS.map((step, i) => {
        const isRL = step.id === "rl"
        const skip = isRL && !rlAvailable
        const done = current > i
        const running = current === i && active
        return (
          <div
            key={step.id}
            className="flex items-start gap-2.5 transition-all duration-300"
            style={{ opacity: skip ? 0.25 : done || running ? 1 : 0.35 }}
          >
            {/* indicator */}
            <div
              className="mt-0.5 w-4 h-4 rounded-full shrink-0 flex items-center justify-center text-[8px] font-bold transition-all duration-500"
              style={{
                background: skip
                  ? "rgba(255,255,255,0.04)"
                  : done
                  ? "rgba(0,255,136,0.2)"
                  : running
                  ? "rgba(0,255,136,0.1)"
                  : "rgba(255,255,255,0.04)",
                border: `1px solid ${
                  skip
                    ? "rgba(255,255,255,0.06)"
                    : done
                    ? "rgba(0,255,136,0.5)"
                    : running
                    ? "rgba(0,255,136,0.3)"
                    : "rgba(255,255,255,0.1)"
                }`,
                boxShadow: running ? "0 0 6px rgba(0,255,136,0.4)" : "none",
              }}
            >
              {done ? (
                <span style={{ color: "#00ff88" }}>✓</span>
              ) : running ? (
                <span
                  className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                  style={{ animation: "pulse 0.8s ease-in-out infinite" }}
                />
              ) : (
                <span className="text-neutral-700">{i + 1}</span>
              )}
            </div>

            <div>
              <div
                className="text-[10px] uppercase tracking-wider transition-colors"
                style={{
                  color: done ? "#00ff88" : running ? "#6ee7b7" : "#404040",
                }}
              >
                {step.label}
                {skip && (
                  <span className="ml-1.5 text-[8px] text-neutral-700 normal-case tracking-normal">skipped</span>
                )}
              </div>
              {(running || done) && !skip && (
                <div className="text-[9px] text-neutral-600 mt-0.5 leading-relaxed">
                  {step.desc}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─── Score bar ─────────────────────────────────────────────────────────────
function ScoreBar({ label, value, color, desc }: { label: string; value: number; color: string; desc: string }) {
  const pct = Math.max(0, Math.min(100, value * 100))
  const [hovered, setHovered] = useState(false)
  return (
    <div
      className="group relative"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="flex items-center gap-2 text-[9px]">
        <span className="text-neutral-600 w-16 uppercase tracking-wider shrink-0">{label}</span>
        <div className="flex-1 h-0.5 bg-white/5 overflow-hidden rounded-full">
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{ width: `${pct}%`, background: color, boxShadow: `0 0 4px ${color}80` }}
          />
        </div>
        <span className="text-neutral-500 w-8 text-right font-mono tabular-nums">{value.toFixed(2)}</span>
      </div>
      {hovered && (
        <div
          className="absolute left-16 -top-6 text-[9px] text-neutral-400 px-2 py-1 rounded-sm z-20 whitespace-nowrap"
          style={{ background: "rgba(0,5,16,0.95)", border: "1px solid rgba(255,255,255,0.08)" }}
        >
          {desc}
        </div>
      )}
    </div>
  )
}

// ─── Star rating ───────────────────────────────────────────────────────────
function StarRating({
  entryId,
  initial,
}: {
  entryId: string | null
  initial?: number | null
}) {
  const [rating, setRating] = useState<number | null>(initial ?? null)
  const [hover, setHover] = useState<number | null>(null)
  const [saved, setSaved] = useState(!!initial)

  if (!entryId) return null

  const submit = async (r: number) => {
    setRating(r)
    setSaved(false)
    try {
      await api.rateRetrieval(entryId, r)
      setSaved(true)
    } catch {}
  }

  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((s) => (
        <button
          key={s}
          onClick={() => submit(s)}
          onMouseEnter={() => setHover(s)}
          onMouseLeave={() => setHover(null)}
          className="text-[13px] transition-all"
          style={{
            color:
              (hover !== null ? s <= hover : s <= (rating ?? 0))
                ? "#f59e0b"
                : "rgba(255,255,255,0.12)",
            textShadow:
              (hover !== null ? s <= hover : s <= (rating ?? 0))
                ? "0 0 6px rgba(245,158,11,0.5)"
                : "none",
          }}
        >
          ★
        </button>
      ))}
      {saved && rating && (
        <span className="ml-1.5 text-[8px] text-amber-600 uppercase tracking-wider">saved</span>
      )}
    </div>
  )
}

// ─── Result card ───────────────────────────────────────────────────────────
function ResultCard({
  r,
  idx,
  rejected = false,
  compact = false,
  entryId: _entryId,
}: {
  r: RetrievedResult
  idx: number
  rejected?: boolean
  compact?: boolean
  entryId?: string | null
}) {
  const color = tierColor(r.tier)
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className="relative rounded-sm overflow-hidden transition-all duration-200 group"
      style={{
        background: "rgba(0,5,16,0.7)",
        border: "1px solid rgba(255,255,255,0.05)",
        borderLeft: `2px solid ${rejected ? "rgba(239,68,68,0.4)" : color}`,
        opacity: rejected ? 0.5 : 1,
      }}
      onMouseEnter={(e) => {
        if (rejected || compact) return
        const el = e.currentTarget as HTMLElement
        el.style.boxShadow = `0 0 24px ${color}18`
        el.style.borderColor = `${color}22`
      }}
      onMouseLeave={(e) => {
        if (rejected || compact) return
        const el = e.currentTarget as HTMLElement
        el.style.boxShadow = "none"
        el.style.borderColor = "rgba(255,255,255,0.05)"
      }}
    >
      {rejected && (
        <div className="absolute top-2 right-2">
          <span
            className="text-[8px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded-sm"
            style={{ color: "#f87171", background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.2)" }}
          >
            REJECTED
          </span>
        </div>
      )}

      <div className={`flex gap-3 ${compact ? "p-2.5" : "p-4"}`}>
        {/* Index */}
        <div
          className={`font-bold tabular-nums shrink-0 leading-none mt-0.5 ${compact ? "text-[13px]" : "text-[20px]"}`}
          style={{ color: "rgba(255,255,255,0.08)" }}
        >
          {String(idx + 1).padStart(2, "0")}
        </div>

        <div className="flex-1 min-w-0 space-y-2">
          {/* Meta row */}
          <div className="flex flex-wrap items-center gap-2">
            <span
              className="text-[8px] uppercase tracking-wider px-1.5 py-0.5 rounded-sm"
              style={{ color, background: `${color}15` }}
            >
              T{r.tier} {r.domain || "—"}
            </span>
            {r.subdomain && (
              <span className="text-[8px] text-neutral-600 uppercase tracking-wider">
                {r.subdomain}
              </span>
            )}
            <span
              className="ml-auto text-[10px] font-mono tabular-nums px-2 py-0.5 rounded-sm shrink-0"
              style={{ color, background: `${color}12`, border: `1px solid ${color}20` }}
            >
              {r.score.toFixed(3)}
            </span>
          </div>

          {/* Text */}
          <p
            className={`text-neutral-300 leading-relaxed ${compact ? "text-[10px] line-clamp-2" : "text-[11px]"} ${!compact && !expanded && "line-clamp-3"}`}
          >
            {r.text}
          </p>

          {!compact && r.text.length > 180 && (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="text-[9px] text-neutral-600 hover:text-neutral-400 uppercase tracking-wider transition-colors"
            >
              {expanded ? "show less ▲" : "show more ▼"}
            </button>
          )}

          {/* Score bars */}
          {!compact && (
            <div className="space-y-1.5 pt-1">
              {SCORE_META.map(({ key, label, desc, color: c }) => (
                <ScoreBar
                  key={key}
                  label={label}
                  value={(r as any)[key] as number}
                  color={c}
                  desc={desc}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Compare view ──────────────────────────────────────────────────────────
interface CompareResult {
  query: string
  query_domain: string
  rl_available: boolean
  hybrid: RetrievedResult[]
  rl: RetrievedResult[]
  overlap_count: number
}

function CompareView({ result }: { result: CompareResult }) {
  const hybridIds = new Set(result.hybrid.map((r) => r.node_id))
  const rlIds = new Set(result.rl.map((r) => r.node_id))

  const statusOf = (id: string) => {
    const inH = hybridIds.has(id)
    const inR = rlIds.has(id)
    if (inH && inR) return "overlap"
    return inH ? "hybrid-only" : "rl-only"
  }

  const overlapPct = Math.round((result.overlap_count / Math.max(result.hybrid.length, 1)) * 100)

  return (
    <div className="space-y-4">
      {/* Stats bar */}
      <div
        className="flex items-center gap-6 px-4 py-3 rounded-sm text-[10px]"
        style={{ background: "rgba(0,5,16,0.5)", border: "1px solid rgba(255,255,255,0.05)" }}
      >
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-amber-400" style={{ boxShadow: "0 0 4px rgba(251,191,36,0.5)" }} />
          <span className="text-neutral-500 uppercase tracking-wider">Domain: <span className="text-amber-400">{result.query_domain || "—"}</span></span>
        </div>
        <div className="h-3 w-px bg-white/10" />
        <div className="flex items-center gap-1.5">
          <span className="text-neutral-500 uppercase tracking-wider">
            Agreement: <span style={{ color: overlapPct >= 60 ? "#00ff88" : "#f59e0b" }}>{result.overlap_count}/{result.hybrid.length} ({overlapPct}%)</span>
          </span>
        </div>
        <div className="ml-auto flex items-center gap-3 text-[9px]">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-emerald-400" style={{ boxShadow: "0 0 4px rgba(0,255,136,0.5)" }} />
            <span className="text-neutral-600">both agreed</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-amber-400" />
            <span className="text-neutral-600">hybrid only</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-violet-400" />
            <span className="text-neutral-600">RL only</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Hybrid column */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-2 h-2 rounded-full bg-amber-400" style={{ boxShadow: "0 0 4px rgba(251,191,36,0.5)" }} />
            <span className="text-[10px] uppercase tracking-wider text-amber-400">Hybrid Scoring</span>
            <span className="text-[9px] text-neutral-700 ml-auto">cosine · recency · domain</span>
          </div>
          <div className="space-y-2">
            {result.hybrid.map((r, idx) => {
              const status = statusOf(r.node_id)
              return (
                <div
                  key={r.node_id}
                  className="relative"
                  style={{
                    borderRadius: "4px",
                    boxShadow: status === "overlap"
                      ? "0 0 10px rgba(0,255,136,0.15)"
                      : "0 0 6px rgba(251,191,36,0.1)",
                    outline: status === "overlap" ? "1px solid rgba(0,255,136,0.15)" : "none",
                  }}
                >
                  {status === "overlap" && (
                    <span
                      className="absolute -left-2 top-1/2 -translate-y-1/2 text-[9px]"
                      style={{ color: "#00ff88" }}
                    >
                      ✦
                    </span>
                  )}
                  <ResultCard r={r} idx={idx} compact />
                </div>
              )
            })}
          </div>
        </div>

        {/* RL column */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <div
              className={`w-2 h-2 rounded-full ${result.rl_available ? "bg-emerald-400" : "bg-neutral-700"}`}
              style={{ boxShadow: result.rl_available ? "0 0 4px rgba(0,255,136,0.5)" : "none" }}
            />
            <span className={`text-[10px] uppercase tracking-wider ${result.rl_available ? "text-emerald-400" : "text-neutral-600"}`}>
              RL Agent {!result.rl_available && "(not available)"}
            </span>
            {result.rl_available && (
              <span className="text-[9px] text-neutral-700 ml-auto">learned policy</span>
            )}
          </div>

          {result.rl_available ? (
            <div className="space-y-2">
              {result.rl.map((r, idx) => {
                const status = statusOf(r.node_id)
                const hybridRank = result.hybrid.findIndex((h) => h.node_id === r.node_id)
                return (
                  <div
                    key={r.node_id}
                    className="relative"
                    style={{
                      borderRadius: "4px",
                      boxShadow: status === "overlap"
                        ? "0 0 10px rgba(0,255,136,0.15)"
                        : "0 0 6px rgba(139,92,246,0.12)",
                      outline: status === "overlap" ? "1px solid rgba(0,255,136,0.15)" : "none",
                    }}
                  >
                    {status === "overlap" && (
                      <span
                        className="absolute -left-2 top-1/2 -translate-y-1/2 text-[9px]"
                        style={{ color: "#00ff88" }}
                      >
                        ✦
                      </span>
                    )}
                    {status === "rl-only" && (
                      <span
                        className="absolute -left-2 top-1/2 -translate-y-1/2 text-[9px]"
                        style={{ color: "#a78bfa" }}
                        title="RL picked this but hybrid didn't"
                      >
                        ◆
                      </span>
                    )}
                    {hybridRank >= 0 && hybridRank !== idx && (
                      <div
                        className="absolute top-1.5 right-2 text-[8px] tabular-nums"
                        style={{ color: hybridRank < idx ? "#f87171" : "#4ade80" }}
                        title={`Hybrid rank: #${hybridRank + 1}`}
                      >
                        {hybridRank < idx ? `▼ was #${hybridRank + 1}` : `▲ was #${hybridRank + 1}`}
                      </div>
                    )}
                    <ResultCard r={r} idx={idx} compact />
                  </div>
                )
              })}
            </div>
          ) : (
            <div
              className="py-12 text-center rounded-sm"
              style={{ border: "1px dashed rgba(255,255,255,0.06)" }}
            >
              <div className="text-[9px] text-neutral-700 uppercase tracking-wider">RL not available</div>
              <div className="text-[9px] text-neutral-800 mt-1">Train the agent in Settings first</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Empty state ───────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 select-none">
      <div className="relative mb-6">
        <div
          className="w-20 h-20 rounded-full flex items-center justify-center"
          style={{ border: "1px solid rgba(0,255,136,0.12)", boxShadow: "0 0 40px rgba(0,255,136,0.06)" }}
        >
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center"
            style={{ border: "1px solid rgba(0,255,136,0.18)", animation: "pulse 3s ease-in-out infinite" }}
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ background: "rgba(0,255,136,0.5)", boxShadow: "0 0 8px rgba(0,255,136,0.6)" }}
            />
          </div>
        </div>
        {[26, 40, 54].map((r, i) => (
          <div
            key={i}
            className="absolute top-1/2 left-1/2 rounded-full"
            style={{
              width: r * 2,
              height: r * 2,
              marginLeft: -r,
              marginTop: -r,
              border: "1px solid rgba(0,255,136,0.06)",
              animation: `spin ${7 + i * 2}s linear infinite`,
              animationDirection: i % 2 ? "reverse" : "normal",
            }}
          />
        ))}
      </div>
      <div className="text-[9px] uppercase tracking-[0.3em] text-neutral-700 mb-1">// STANDBY</div>
      <div className="text-[11px] text-neutral-600 uppercase tracking-widest" style={{ animation: "pulse 2.5s ease-in-out infinite" }}>
        Awaiting query
      </div>
    </div>
  )
}

// ─── Main page ─────────────────────────────────────────────────────────────
export function RetrievalPage() {
  const [query, setQuery]                   = useState("")
  const [result, setResult]                 = useState<RetrieveResponse | null>(null)
  const [loading, setLoading]               = useState(false)
  const [rlStatus, setRlStatus]              = useState<RLStatus | null>(null)
  const [mode, setMode]                     = useState<"single" | "compare">("single")
  const [compareResult, setCompareResult]   = useState<CompareResult | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [showRejected, setShowRejected]     = useState(false)
  const [includeRejected, setIncludeRejected] = useState(false)
  const [queryHistory, setQueryHistory]     = useState<string[]>([])
  const inputRef = useRef<HTMLInputElement>(null)
  const { colors } = useTheme()

  const isLoading = loading || compareLoading

  const refreshRlStatus = useCallback(() => {
    api.rlStatus().then(setRlStatus).catch(() => {})
  }, [])

  useEffect(() => {
    refreshRlStatus()
    inputRef.current?.focus()
  }, [])

  const run = async () => {
    if (!query.trim() || isLoading) return
    setLoading(true)
    setResult(null)
    setCompareResult(null)
    try {
      const res = await api.retrieve(query.trim(), 10, includeRejected)
      setResult(res)
      setQueryHistory((h) => [query.trim(), ...h.filter((q) => q !== query.trim())].slice(0, 8))
    } finally {
      setLoading(false)
    }
  }

  const runCompare = async () => {
    if (!query.trim() || isLoading) return
    setCompareLoading(true)
    setResult(null)
    setCompareResult(null)
    try {
      const res = await api.compareRetrieve(query.trim(), 5)
      setCompareResult(res)
      setQueryHistory((h) => [query.trim(), ...h.filter((q) => q !== query.trim())].slice(0, 8))
    } finally {
      setCompareLoading(false)
    }
  }

  const submit = () => (mode === "compare" ? runCompare() : run())

  return (
    <div className="relative h-full min-h-0 font-mono overflow-y-auto bg-[#020d0d] custom-scrollbar">
      <ScanlineOverlay />

      {/* ambient glow */}
      <div
        className="pointer-events-none fixed inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 0%, rgba(0,255,136,0.06) 0%, transparent 55%), radial-gradient(ellipse at 50% 100%, rgba(0,229,255,0.04) 0%, transparent 55%)",
        }}
      />

      {/* corner brackets */}
      {[
        "top-3 left-3 border-t-2 border-l-2",
        "top-3 right-3 border-t-2 border-r-2",
        "bottom-3 left-3 border-b-2 border-l-2",
        "bottom-3 right-3 border-b-2 border-r-2",
      ].map((cls) => (
        <div
          key={cls}
          className={`pointer-events-none absolute h-5 w-5 border-emerald-400/30 ${cls}`}
          style={{ filter: "drop-shadow(0 0 4px rgba(0,255,136,0.4))" }}
        />
      ))}

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-8">
        {/* ── Header ── */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px w-8" style={{ background: colors.primaryBorder }} />
            <span className="text-[9px] uppercase tracking-[0.3em]" style={{ color: colors.primaryTextDim }}>Memory Retrieval</span>
          </div>
          <div className="flex items-end gap-4">
            <h1
              className="text-xl font-bold tracking-wide"
              style={{ color: colors.primaryText, textShadow: `0 0 20px ${colors.primaryGlow}` }}
            >
              Query Engine
            </h1>
            {rlStatus && (
              <span
                className="mb-0.5 text-[9px] uppercase tracking-widest px-2 py-0.5 rounded-sm"
                style={{
                  color: rlStatus.available ? "#00ff88" : rlStatus.enabled ? "#f59e0b" : "#525252",
                  background: rlStatus.available
                    ? "rgba(0,255,136,0.08)"
                    : rlStatus.enabled
                    ? "rgba(245,158,11,0.08)"
                    : "rgba(255,255,255,0.04)",
                  border: `1px solid ${rlStatus.available ? "rgba(0,255,136,0.2)" : rlStatus.enabled ? "rgba(245,158,11,0.2)" : "rgba(255,255,255,0.08)"}`,
                }}
              >
                {rlStatus.available ? "RL Active" : rlStatus.enabled ? "RL Loading" : "Hybrid Mode"}
              </span>
            )}
          </div>
        </div>

        {/* ── Query bar ── */}
        <div
          className="relative mb-6 rounded-sm"
          style={{
            background: "rgba(0,5,16,0.5)",
            border: `1px solid ${isLoading ? "rgba(0,255,136,0.25)" : "rgba(255,255,255,0.07)"}`,
            boxShadow: isLoading ? "0 0 20px rgba(0,255,136,0.06)" : "none",
            transition: "box-shadow 0.3s ease, border-color 0.3s ease",
          }}
        >
          {/* animated left bar */}
          <div
            className="absolute left-0 top-0 bottom-0 w-0.5 rounded-l-sm transition-all duration-300"
            style={{
              background: isLoading
                ? "linear-gradient(180deg, transparent, #22d3ee, #a855f7, transparent)"
                : "linear-gradient(180deg, transparent, rgba(0,255,136,0.4), transparent)",
              boxShadow: isLoading ? "0 0 8px rgba(0,229,255,0.5)" : "none",
              animation: isLoading ? "scanPulse 1.2s ease-in-out infinite" : "none",
            }}
          />

          <div className="flex items-center">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder="QUERY THE MEMORY GRAPH..."
              disabled={isLoading}
              className="flex-1 bg-transparent py-4 pl-4 pr-4 text-base text-neutral-100 placeholder:text-neutral-700 focus:outline-none disabled:opacity-60"
              style={{ caretColor: "#22d3ee" }}
            />

            {/* mode toggle */}
            <div className="flex items-center gap-1 px-3 border-l border-white/5">
              {(["single", "compare"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  disabled={m === "compare" && !rlStatus?.available}
                  className="px-3 py-1.5 text-[9px] uppercase tracking-wider rounded-sm transition-all disabled:opacity-25"
                  style={{
                    color: mode === m ? "#6ee7b7" : "#404040",
                    background: mode === m ? "rgba(0,255,136,0.08)" : "transparent",
                    border: mode === m ? "1px solid rgba(0,255,136,0.2)" : "1px solid transparent",
                  }}
                >
                  {m}
                </button>
              ))}
            </div>

            {/* rejected toggle (single + RL active) */}
            {mode === "single" && rlStatus?.available && (
              <div className="px-3 border-l border-white/5">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeRejected}
                    onChange={(e) => setIncludeRejected(e.target.checked)}
                    className="sr-only"
                  />
                  <div
                    className="w-8 h-4 rounded-full transition-colors relative"
                    style={{
                      background: includeRejected ? "rgba(0,255,136,0.25)" : "rgba(255,255,255,0.05)",
                      border: "1px solid rgba(0,255,136,0.25)",
                    }}
                  >
                    <div
                      className="absolute top-0.5 w-3 h-3 rounded-full transition-transform"
                      style={{
                        background: includeRejected ? "#00ff88" : "#404040",
                        left: includeRejected ? "calc(100% - 14px)" : "2px",
                        boxShadow: includeRejected ? "0 0 4px rgba(0,255,136,0.6)" : "none",
                      }}
                    />
                  </div>
                  <span className="text-[9px] uppercase tracking-wider text-neutral-600">Show rejected</span>
                </label>
              </div>
            )}

            {/* scan button */}
            <button
              onClick={submit}
              disabled={isLoading || !query.trim()}
              className="px-5 py-4 text-[10px] uppercase tracking-widest border-l transition-all disabled:opacity-30"
              style={{
                borderColor: "rgba(0,255,136,0.2)",
                color: "#00ff88",
                background: isLoading ? "rgba(0,255,136,0.04)" : "transparent",
              }}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <span className="inline-flex gap-0.5">
                    {[0, 150, 300].map((d) => (
                      <span
                        key={d}
                        className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce"
                        style={{ animationDelay: `${d}ms` }}
                      />
                    ))}
                  </span>
                  Scanning
                </span>
              ) : (
                "[ SCAN ]"
              )}
            </button>
          </div>
        </div>

        {/* ── Query history pills ── */}
        {queryHistory.length > 0 && !isLoading && (
          <div className="flex flex-wrap gap-1.5 mb-5">
            {queryHistory.map((q) => (
              <button
                key={q}
                onClick={() => { setQuery(q); setTimeout(submit, 0) }}
                className="text-[9px] px-2.5 py-1 rounded-sm text-neutral-600 hover:text-neutral-300 transition-colors"
                style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
              >
                {q.length > 40 ? q.slice(0, 40) + "…" : q}
              </button>
            ))}
          </div>
        )}

        {/* ── Body: sidebar + results ── */}
        <div className="flex gap-5">
          {/* Left sidebar */}
          <div className="w-56 shrink-0 space-y-4">
            {/* Pipeline steps */}
            <div
              className="rounded-sm p-4"
              style={{ background: "rgba(0,5,16,0.7)", border: "1px solid rgba(255,255,255,0.06)" }}
            >
              <div className="text-[8px] uppercase tracking-widest text-neutral-700 mb-3">
                // Retrieval Pipeline
              </div>
              <PipelineSteps
                active={isLoading}
                rlAvailable={rlStatus?.available ?? false}
              />
            </div>

          </div>

          {/* Results area */}
          <div className="flex-1 min-w-0">

            {/* Single mode results */}
            {mode === "single" && result && (
              <div className="space-y-4">
                {/* Results meta bar */}
                <div
                  className="flex items-center gap-4 px-4 py-2.5 rounded-sm text-[10px]"
                  style={{ background: "rgba(0,5,16,0.5)", border: "1px solid rgba(255,255,255,0.05)" }}
                >
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400"
                      style={{ boxShadow: "0 0 4px rgba(0,255,136,0.8)" }} />
                    <span className="text-neutral-500 uppercase tracking-wider">
                      Domain: <span className="text-emerald-400">{result.query_domain || "—"}</span>
                    </span>
                  </div>
                  <div className="h-3 w-px bg-white/10" />
                  <span className="text-neutral-500 uppercase tracking-wider">
                    Results: <span className="text-neutral-300">{result.results.length}</span>
                  </span>
                  {result.rejected && result.rejected.length > 0 && (
                    <>
                      <div className="h-3 w-px bg-white/10" />
                      <span className="text-neutral-500 uppercase tracking-wider">
                        Rejected by RL: <span className="text-red-400">{result.rejected.length}</span>
                      </span>
                    </>
                  )}
                  {result.entry_id && (
                    <div className="ml-auto">
                      <StarRating entryId={result.entry_id} />
                    </div>
                  )}
                </div>

                {/* Result cards */}
                <div className="space-y-3">
                  {result.results.map((r, idx) => (
                    <ResultCard
                      key={r.node_id}
                      r={r}
                      idx={idx}
                      entryId={result.entry_id}
                    />
                  ))}
                </div>

                {/* Rejected section */}
                {result.rejected && result.rejected.length > 0 && (
                  <div className="mt-4">
                    <button
                      onClick={() => setShowRejected((v) => !v)}
                      className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-neutral-600 hover:text-neutral-400 transition-colors mb-3"
                    >
                      <span style={{ transform: showRejected ? "rotate(90deg)" : "none", display: "inline-block" }}>▶</span>
                      RL Rejected Candidates ({result.rejected.length})
                      <span className="text-[8px] text-neutral-700 normal-case tracking-normal">
                        — the RL agent considered these but chose not to return them
                      </span>
                    </button>
                    {showRejected && (
                      <div className="space-y-2">
                        {result.rejected.map((r, idx) => (
                          <ResultCard key={r.node_id} r={r} idx={idx} rejected />
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Compare mode results */}
            {mode === "compare" && compareResult && (
              <CompareView result={compareResult} />
            )}

            {/* Empty state */}
            {!result && !compareResult && !isLoading && <EmptyState />}

            {/* Loading placeholder */}
            {isLoading && !result && !compareResult && (
              <div className="space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div
                    key={i}
                    className="rounded-sm h-20"
                    style={{
                      background: "rgba(0,5,16,0.5)",
                      border: "1px solid rgba(255,255,255,0.04)",
                      borderLeft: "2px solid rgba(0,255,136,0.08)",
                      animation: `pulse 1.5s ease-in-out ${i * 0.1}s infinite`,
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes scanPulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  )
}
