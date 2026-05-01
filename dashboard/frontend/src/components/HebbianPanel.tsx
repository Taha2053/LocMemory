import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { HudPanel, StatusDot } from "@/components/hud"

interface HebbianStats {
  count: number
  min_weight: number
  max_weight: number
  avg_weight: number
  histogram: { range: string; count: number }[]
  active_edges?: number   // neurons that fire together (weight >= 0.8)
  strong_edges?: number   // strong connections (weight >= 1.0)
}

export function HebbianPanel() {
  const [data, setData] = useState<HebbianStats | null>(null)
  const [decaying, setDecaying] = useState(false)
  const [flash, setFlash] = useState<string | null>(null)

  const load = () => api.hebbianStats().then(setData).catch(() => {})

  useEffect(() => { load() }, [])

  const decay = async () => {
    setDecaying(true)
    setFlash(null)
    try {
      const res = await api.hebbianDecay()
      setFlash(`${res.edges_decayed} edges decayed`)
      await load()
    } catch {
      setFlash("decay failed")
    } finally {
      setDecaying(false)
      setTimeout(() => setFlash(null), 3000)
    }
  }

  const maxCount = data
    ? Math.max(1, ...data.histogram.map((b) => b.count))
    : 1

  return (
    <HudPanel id="SYS.HBB.06" className="hud-panel" progressValue={data ? 65 : 20}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] uppercase tracking-widest text-neutral-400">
          Edge Dynamics
        </div>
        <StatusDot label="LIVE" color="#009b94" />
      </div>

      {!data ? (
        <div className="text-[10px] text-neutral-600 italic">loading...</div>
      ) : (
        <>
          {/* Weight histogram */}
          <div className="mb-2">
            <div className="text-[8px] uppercase tracking-wider text-neutral-600 mb-1">
              weight distribution
            </div>
            <div className="flex items-end gap-px h-8">
              {data.histogram.map((b, i) => {
                const pct = (b.count / maxCount) * 100
                return (
                  <div
                    key={i}
                    className="flex-1 transition-all duration-500"
                    style={{
                      height: `${Math.max(pct, 2)}%`,
                      background: `rgba(0, 255, 136,${0.3 + (pct / 100) * 0.6})`,
                    }}
                    title={`${b.range}: ${b.count}`}
                  />
                )
              })}
            </div>
            <div className="flex justify-between text-[8px] text-neutral-600 mt-0.5">
              <span>0.0</span>
              <span>weight →</span>
              <span>5.0</span>
            </div>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-4 gap-1 mb-2">
            {[
              { label: "edges", value: data.count },
              { label: "avg w", value: (data.avg_weight ?? 0).toFixed(2) },
              { label: "max w", value: (data.max_weight ?? 0).toFixed(2) },
              { label: "active", value: data.active_edges ?? 0 },  // "neurons that fire together"
            ].map(({ label, value }) => (
              <div key={label} className="border border-emerald-400/10 bg-black/30 px-1.5 py-1 text-center">
                <div className="text-[8px] uppercase tracking-wider text-neutral-500">{label}</div>
                <div className="text-[11px] font-mono text-neutral-100 tabular-nums">{value}</div>
              </div>
            ))}
          </div>

          {flash && (
            <div className="text-[10px] text-emerald-400/80 mb-1.5">{flash}</div>
          )}

          <button
            onClick={decay}
            disabled={decaying}
            className="scan-button border border-emerald-400/30 px-3 py-1 text-[10px] text-emerald-400/70 uppercase tracking-wider transition-all disabled:opacity-40 w-full"
          >
            {decaying ? "decaying..." : "[ apply decay ]"}
          </button>
        </>
      )}
    </HudPanel>
  )
}
