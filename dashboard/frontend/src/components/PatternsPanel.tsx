import { useEffect, useState, useCallback } from "react"
import { api } from "@/lib/api"
import { HudPanel, StatusDot } from "@/components/hud"

interface Pattern {
  id: string
  text: string
  domain: string
  created_at: string
}

const REFRESH_INTERVAL = 15_000

export function PatternsPanel() {
  const [patterns, setPatterns]         = useState<Pattern[]>([])
  const [error, setError]               = useState(false)
  const [detecting, setDetecting]       = useState(false)
  const [consolidating, setConsolidating] = useState(false)
  const [flash, setFlash]               = useState<string | null>(null)
  const [flashOk, setFlashOk]           = useState(true)

  const loadPatterns = useCallback(() => {
    api.patterns()
      .then((d) => { setPatterns(d); setError(false) })
      .catch(() => setError(true))
  }, [])

  useEffect(() => {
    loadPatterns()
    const id = setInterval(loadPatterns, REFRESH_INTERVAL)
    return () => clearInterval(id)
  }, [loadPatterns])

  const showFlash = (msg: string, ok = true) => {
    setFlash(msg)
    setFlashOk(ok)
    setTimeout(() => setFlash(null), 5000)
  }

  const detect = async () => {
    setDetecting(true)
    try {
      const res = await api.detectPatterns()
      showFlash(
        res.procedural_nodes_created > 0
          ? `+${res.procedural_nodes_created} new patterns detected`
          : "no new patterns found",
        res.procedural_nodes_created > 0,
      )
      loadPatterns()
    } catch {
      showFlash("detection failed — is backend running?", false)
    } finally {
      setDetecting(false)
    }
  }

  const consolidate = async () => {
    setConsolidating(true)
    try {
      const res = await api.consolidate()
      showFlash(
        `${res.clusters_found} clusters → ${res.anchors_created} anchors`,
        res.anchors_created > 0,
      )
      loadPatterns()
    } catch {
      showFlash("consolidation failed — is backend running?", false)
    } finally {
      setConsolidating(false)
    }
  }

  const busy = detecting || consolidating

  const dotColor = error
    ? "#ef4444"
    : patterns.length > 0 ? "#a855f7" : "#525252"
  const dotLabel = error
    ? "ERROR"
    : patterns.length > 0 ? "ACTIVE" : "IDLE"

  return (
    <HudPanel id="Pattern Detection" className="hud-panel p-4" progressValue={patterns.length > 0 ? 88 : 30}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="text-[10px] uppercase tracking-widest text-neutral-400">
          Cognitive Ops
        </div>
        <StatusDot label={dotLabel} color={dotColor} />
      </div>

      {error ? (
        <div className="text-[10px] text-red-400/70 italic py-2">
          backend unreachable — start the server
        </div>
      ) : (
        <>
          {/* Pattern count */}
          <div className="flex items-center justify-between text-[10px] mb-2">
            <span className="text-neutral-500 uppercase tracking-wider">
              procedural patterns
            </span>
            <span className="font-mono tabular-nums" style={{ color: dotColor }}>
              {patterns.length}
            </span>
          </div>

          {/* Pattern list */}
          <div className="space-y-1 mb-3 max-h-[90px] overflow-y-auto pr-0.5">
            {patterns.length === 0 ? (
              <div className="text-[10px] text-neutral-600 italic">
                none detected yet — click detect
              </div>
            ) : (
              patterns.slice(0, 5).map((p) => (
                <div
                  key={p.id}
                  className="flex items-start gap-1.5 py-0.5 border-l-2 border-purple-400/25 pl-1.5"
                >
                  <span className="inline-block h-1.5 w-1.5 mt-[3px] shrink-0 rounded-full bg-purple-400/60" />
                  <span
                    className="text-[10px] text-neutral-300 leading-tight"
                    style={{
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                    }}
                  >
                    {p.text}
                  </span>
                </div>
              ))
            )}
          </div>

          {/* Actions */}
          <div className="border-t border-white/5 pt-2 space-y-1.5">
            {flash && (
              <div className={`text-[10px] mb-1 ${flashOk ? "text-purple-400/80" : "text-red-400/80"}`}>
                {flash}
              </div>
            )}
            <button
              onClick={detect}
              disabled={busy}
              className="scan-button w-full border border-purple-400/30 px-2 py-1.5 text-[10px] text-purple-400/70 uppercase tracking-wider transition-all disabled:opacity-40 hover:border-purple-400/60 hover:text-purple-400 whitespace-nowrap"
            >
              {detecting ? "detecting..." : "[ detect patterns ]"}
            </button>
            <button
              onClick={consolidate}
              disabled={busy}
              className="scan-button w-full border border-emerald-400/30 px-2 py-1.5 text-[10px] text-emerald-400/70 uppercase tracking-wider transition-all disabled:opacity-40 hover:border-emerald-400/60 hover:text-emerald-400 whitespace-nowrap"
            >
              {consolidating ? "consolidating..." : "[ consolidate ]"}
            </button>
          </div>
        </>
      )}
    </HudPanel>
  )
}
