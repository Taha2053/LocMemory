import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { HudPanel, StatusDot } from "@/components/hud"

interface Pattern {
  id: string
  text: string
  domain: string
  created_at: string
}

export function PatternsPanel() {
  const [patterns, setPatterns] = useState<Pattern[]>([])
  const [detecting, setDetecting] = useState(false)
  const [consolidating, setConsolidating] = useState(false)
  const [flash, setFlash] = useState<string | null>(null)

  const loadPatterns = () =>
    api.patterns().then(setPatterns).catch(() => {})

  useEffect(() => { loadPatterns() }, [])

  const detect = async () => {
    setDetecting(true)
    setFlash(null)
    try {
      const res = await api.detectPatterns()
      setFlash(`+${res.procedural_nodes_created} patterns`)
      await loadPatterns()
    } catch {
      setFlash("detection failed")
    } finally {
      setDetecting(false)
      setTimeout(() => setFlash(null), 3000)
    }
  }

  const consolidate = async () => {
    setConsolidating(true)
    setFlash(null)
    try {
      const res = await api.consolidate()
      setFlash(`${res.anchors_created} anchors created`)
    } catch {
      setFlash("consolidation failed")
    } finally {
      setConsolidating(false)
      setTimeout(() => setFlash(null), 3000)
    }
  }

  return (
    <HudPanel id="SYS.PCT.07" className="hud-panel" progressValue={patterns.length > 0 ? 88 : 30}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] uppercase tracking-widest text-neutral-400">
          Cognitive Ops
        </div>
        <StatusDot
          label={patterns.length > 0 ? "ACTIVE" : "IDLE"}
          color={patterns.length > 0 ? "#a855f7" : "#525252"}
        />
      </div>

      {/* Pattern count + recent list */}
      <div className="mb-2">
        <div className="flex items-center justify-between text-[10px] mb-1.5">
          <span className="text-neutral-500 uppercase tracking-wider">
            procedural patterns
          </span>
          <span className="font-mono text-purple-400/90 tabular-nums">
            {patterns.length}
          </span>
        </div>
        <div className="space-y-1 max-h-[80px] overflow-y-auto pr-0.5">
          {patterns.length === 0 ? (
            <div className="text-[10px] text-neutral-600 italic">
              none detected yet
            </div>
          ) : (
            patterns.slice(0, 4).map((p) => (
              <div
                key={p.id}
                className="flex items-start gap-1.5 py-0.5 border-l-2 border-purple-400/20 pl-1.5"
              >
                <span className="inline-block h-1.5 w-1.5 mt-1 shrink-0 rounded-full bg-purple-400/60" />
                <span className="text-[10px] text-neutral-300 leading-tight line-clamp-1">
                  {p.text}
                </span>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="border-t border-white/5 pt-2 space-y-1">
        {flash && (
          <div className="text-[10px] text-purple-400/80 mb-1">{flash}</div>
        )}
        <div className="flex gap-1.5">
          <button
            onClick={detect}
            disabled={detecting || consolidating}
            className="scan-button flex-1 border border-purple-400/30 px-2 py-1 text-[10px] text-purple-400/70 uppercase tracking-wider transition-all disabled:opacity-40"
          >
            {detecting ? "detecting..." : "[ detect ]"}
          </button>
          <button
            onClick={consolidate}
            disabled={detecting || consolidating}
            className="scan-button flex-1 border border-emerald-400/30 px-2 py-1 text-[10px] text-emerald-400/70 uppercase tracking-wider transition-all disabled:opacity-40"
          >
            {consolidating ? "running..." : "[ consolidate ]"}
          </button>
        </div>
      </div>
    </HudPanel>
  )
}
