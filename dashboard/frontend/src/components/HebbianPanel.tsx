import { useEffect, useState, useCallback } from "react"
import { api } from "@/lib/api"
import { HudPanel, StatusDot } from "@/components/hud"

interface HebbianStats {
  count: number
  min_weight: number
  max_weight: number
  avg_weight: number
  active_edges?: number
  strong_edges?: number
  histogram: { range: string; count: number }[]
}

const REFRESH_INTERVAL = 12_000

export function HebbianPanel() {
  const [data, setData]       = useState<HebbianStats | null>(null)
  const [error, setError]     = useState(false)
  const [decaying, setDecaying] = useState(false)
  const [flash, setFlash]     = useState<string | null>(null)

  const load = useCallback(() => {
    api.hebbianStats()
      .then((d) => { setData(d); setError(false) })
      .catch(() => setError(true))
  }, [])

  useEffect(() => {
    load()
    const id = setInterval(load, REFRESH_INTERVAL)
    return () => clearInterval(id)
  }, [load])

  const decay = async () => {
    setDecaying(true)
    setFlash(null)
    try {
      const res = await api.hebbianDecay()
      setFlash(res.edges_decayed > 0
        ? `${res.edges_decayed} edges decayed`
        : "no edges to decay"
      )
      load()
    } catch {
      setFlash("decay failed — is backend running?")
    } finally {
      setDecaying(false)
      setTimeout(() => setFlash(null), 4000)
    }
  }

  const maxCount = data ? Math.max(1, ...data.histogram.map((b) => b.count)) : 1

  const dotColor  = error ? "#ef4444" : "#009b94"
  const dotLabel  = error ? "ERROR"   : data ? "LIVE" : "LOADING"

  return (
    <HudPanel id="Hebbian Learning" className="hud-panel p-4" progressValue={data ? 65 : 20}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="text-[10px] uppercase tracking-widest text-neutral-400">
          Edge Dynamics
        </div>
        <StatusDot label={dotLabel} color={dotColor} />
      </div>

      {error ? (
        <div className="text-[10px] text-red-400/70 italic py-2">
          backend unreachable — start the server
        </div>
      ) : !data ? (
        <div className="text-[10px] text-neutral-600 italic">loading...</div>
      ) : (
        <>
          {/* Weight histogram */}
          <div className="mb-3">
            <div className="text-[8px] uppercase tracking-wider text-neutral-600 mb-1.5">
              weight distribution
            </div>
            <div className="flex items-end gap-px h-10">
              {data.histogram.map((b, i) => {
                const pct = (b.count / maxCount) * 100
                const active = b.count > 0
                return (
                  <div
                    key={i}
                    className="flex-1 transition-all duration-700"
                    style={{
                      height: `${Math.max(active ? pct : 0, active ? 4 : 1)}%`,
                      background: active
                        ? `rgba(0,255,136,${0.25 + (pct / 100) * 0.65})`
                        : "rgba(255,255,255,0.04)",
                    }}
                    title={`${b.range}: ${b.count} edges`}
                  />
                )
              })}
            </div>
            <div className="flex justify-between text-[8px] text-neutral-600 mt-1">
              <span>0.0</span>
              <span className="text-neutral-700">weight →</span>
              <span>5.0</span>
            </div>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-4 gap-1 mb-3">
            {[
              { label: "edges",  value: data.count },
              { label: "avg w",  value: (data.avg_weight ?? 0).toFixed(2) },
              { label: "max w",  value: (data.max_weight ?? 0).toFixed(2) },
              { label: "active", value: data.active_edges ?? 0 },
            ].map(({ label, value }) => (
              <div
                key={label}
                className="border border-emerald-400/10 bg-black/30 px-1 py-1.5 text-center"
              >
                <div className="text-[8px] uppercase tracking-wider text-neutral-500 mb-0.5">
                  {label}
                </div>
                <div className="text-[11px] font-mono text-neutral-100 tabular-nums">
                  {value}
                </div>
              </div>
            ))}
          </div>

          {flash && (
            <div className={`text-[10px] mb-2 ${flash.includes("fail") || flash.includes("error") || flash.includes("backend") ? "text-red-400/80" : "text-emerald-400/80"}`}>
              {flash}
            </div>
          )}

          <button
            onClick={decay}
            disabled={decaying}
            className="scan-button w-full border border-emerald-400/30 px-3 py-1.5 text-[10px] text-emerald-400/70 uppercase tracking-wider transition-all disabled:opacity-40 hover:border-emerald-400/60 hover:text-emerald-400"
          >
            {decaying ? "decaying..." : "[ apply decay ]"}
          </button>
        </>
      )}
    </HudPanel>
  )
}
