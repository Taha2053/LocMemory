import { useState } from "react"
import { api, type RetrievedResult } from "@/lib/api"
import { HudPanel } from "@/components/hud"

const TIER_COLORS = ["#00c4bc", "#ff8c26", "#ffd700", "#ff4d6d"]

interface Props {
  onSelect: (id: string) => void
}

export function RetrievalConsole({ onSelect }: Props) {
  const [query, setQuery] = useState("")
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<RetrievedResult[]>([])
  const [queryDomain, setQueryDomain] = useState<string>("")
  const [expanded, setExpanded] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setExpanded(true)
    try {
      const res = await api.retrieve(query.trim(), 8)
      setResults(res.results)
      setQueryDomain(res.query_domain)
    } catch (err) {
      setError(err instanceof Error ? err.message : "retrieval failed")
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="absolute bottom-16 left-1/2 -translate-x-1/2 z-20 w-[640px] max-w-[90vw] pointer-events-auto">
      <HudPanel id="SYS.RTR.04" className="hud-panel" progressValue={loading ? 60 : 100}>
        <div className="flex items-center justify-between mb-2">
          <div className="text-[10px] uppercase tracking-widest text-neutral-400">
            Retrieval Console
          </div>
          {queryDomain && (
            <div className="text-[10px] text-emerald-400/80 uppercase tracking-wider">
              domain: {queryDomain}
            </div>
          )}
        </div>

        <form onSubmit={submit} className="flex items-center gap-2">
          <span className="text-emerald-400/60 text-[12px] font-mono">{">"}</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="query the cognitive graph..."
            className={`flex-1 bg-transparent border-b border-emerald-400/20 focus:border-emerald-400/60 outline-none text-[13px] text-neutral-100 py-1.5 placeholder:text-neutral-600 font-mono ${loading ? "scanline-input" : ""}`}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="scan-button border border-emerald-400/40 px-3 py-1 text-[10px] text-emerald-400/70 uppercase tracking-wider transition-all"
          >
            {loading ? "scanning..." : "[ scan ]"}
          </button>
          {expanded && (
            <button
              type="button"
              onClick={() => { setExpanded(false); setResults([]); setQuery("") }}
              className="border border-neutral-700 px-2 py-1 text-[10px] text-neutral-500 hover:text-neutral-300 transition"
            >
              ×
            </button>
          )}
        </form>

        {error && (
          <div className="mt-2 text-[11px] text-red-400/80">{error}</div>
        )}

        {expanded && results.length > 0 && (
          <div className="mt-3 max-h-[280px] overflow-y-auto space-y-1.5 pr-1">
            {results.map((r, i) => (
              <button
                key={r.node_id}
                onClick={() => onSelect(r.node_id)}
                className="log-entry w-full text-left p-2 border border-emerald-400/10 hover:border-emerald-400/40 hover:bg-emerald-400/5 transition group"
                style={{ animationDelay: `${i * 0.04}s` }}
              >
                <div className="flex items-start gap-2 mb-1.5">
                  <span
                    className="mt-1 inline-block h-2 w-2 shrink-0 rounded-full"
                    style={{
                      backgroundColor: TIER_COLORS[r.tier - 1] || "#888",
                      boxShadow: `0 0 6px ${TIER_COLORS[r.tier - 1] || "#888"}`,
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-[12px] text-neutral-100 line-clamp-2">{r.text}</div>
                    <div className="mt-1 flex items-center gap-3 text-[9px] text-neutral-500 uppercase tracking-wider">
                      <span>{r.domain}/{r.subdomain}</span>
                      <span>depth {r.depth}</span>
                      <span className="text-emerald-400/80 tabular-nums">score {r.score.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
                <ScoreBar
                  cosine={r.cosine_contribution}
                  recency={r.recency_contribution}
                  category={r.category_contribution}
                />
              </button>
            ))}
          </div>
        )}

        {expanded && !loading && results.length === 0 && !error && (
          <div className="mt-3 text-[11px] text-neutral-500 italic">no matches</div>
        )}
      </HudPanel>
    </div>
  )
}

function ScoreBar({ cosine, recency, category }: { cosine: number; recency: number; category: number }) {
  const total = cosine + recency + category || 1
  const c = (cosine / total) * 100
  const r = (recency / total) * 100
  const k = (category / total) * 100
  return (
    <div>
      <div className="flex h-1 w-full overflow-hidden rounded-sm bg-neutral-800/60">
        <div style={{ width: `${c}%`, background: "#00c4bc" }} title={`semantic ${c.toFixed(0)}%`} />
        <div style={{ width: `${r}%`, background: "#ff8c26" }} title={`recency ${r.toFixed(0)}%`} />
        <div style={{ width: `${k}%`, background: "#ffd700" }} title={`category ${k.toFixed(0)}%`} />
      </div>
      <div className="mt-1 flex gap-3 text-[8px] text-neutral-500 uppercase tracking-wider">
        <span><span style={{ color: "#00c4bc" }}>●</span> sem {c.toFixed(0)}%</span>
        <span><span style={{ color: "#ff8c26" }}>●</span> rec {r.toFixed(0)}%</span>
        <span><span style={{ color: "#ffd700" }}>●</span> cat {k.toFixed(0)}%</span>
      </div>
    </div>
  )
}
