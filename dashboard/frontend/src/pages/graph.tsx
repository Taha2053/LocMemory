import { useState } from "react"
import { BrainScene } from "@/components/BrainScene"

const TIER_COLORS = ["#3b82f6", "#06b6d4", "#9ec5e8", "#a855f7"] as const
const TIER_LABELS = [
  "Core Context",
  "Anchor Memories",
  "Leaf Memories",
  "Procedural Memories",
]
const TIER_SUB = ["(Deep)", "(Mid)", "(Facts)", ""]
const TIER_DESC = [
  "High-level semantic hubs",
  "Stable reference points",
  "Atomic facts & observations",
  "Skills, processes & workflows",
]

// Placeholder counts — wire to /api/stats once backend is up
const TIER_COUNTS = [1435, 875, 770, 420]
const TOTAL_NODES = TIER_COUNTS.reduce((a, b) => a + b, 0)
const TOTAL_EDGES = 6681

export function GraphPage() {
  const [, setSelected] = useState<string | null>(null)

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#000510]">
      {/* Brain animation as full-screen background */}
      <BrainScene
        className="absolute inset-0 h-full w-full"
        style={{ width: "100%", height: "100%" }}
      />

      {/* Top-left brand */}
      <div className="pointer-events-none absolute top-5 left-5 z-10">
        <h1 className="text-2xl font-semibold tracking-tight text-cyan-300">
          LocMemory
        </h1>
        <p className="text-[11px] uppercase tracking-[0.2em] text-cyan-500/70">
          cognitive memory graph
        </p>
      </div>

      {/* Memory Tiers panel */}
      <div className="absolute top-5 right-5 z-10 w-72 rounded-lg border border-white/10 bg-black/60 backdrop-blur-md p-4 shadow-2xl">
        <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-400 mb-3">
          Memory Tiers
        </div>
        <div className="space-y-2.5">
          {TIER_LABELS.map((label, i) => (
            <div key={i} className="flex items-start gap-2.5">
              <span
                className="mt-1.5 inline-block h-2 w-2 rounded-full shrink-0"
                style={{
                  backgroundColor: TIER_COLORS[i],
                  boxShadow: `0 0 8px ${TIER_COLORS[i]}`,
                }}
              />
              <div className="flex-1 min-w-0">
                <div className="text-[12px] text-neutral-100 font-medium">
                  {label}{" "}
                  <span className="text-neutral-500 font-normal">
                    {TIER_SUB[i]}
                  </span>
                </div>
                <div className="text-[10px] text-neutral-500 leading-snug">
                  {TIER_DESC[i]}
                </div>
              </div>
              <div className="text-[11px] text-neutral-300 font-mono tabular-nums">
                {TIER_COUNTS[i].toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Graph Metrics panel */}
      <div className="absolute top-[340px] right-5 z-10 w-72 rounded-lg border border-white/10 bg-black/60 backdrop-blur-md p-4 shadow-2xl">
        <div className="text-[10px] uppercase tracking-[0.2em] text-neutral-400 mb-3">
          Graph Metrics
        </div>
        <div className="space-y-1.5 text-[12px]">
          <Metric label="Total Nodes" value={TOTAL_NODES.toLocaleString()} />
          <Metric label="Total Edges" value={TOTAL_EDGES.toLocaleString()} />
          <Metric label="Communities" value="—" />
          <Metric label="Density" value="1.09e-3" />
          <Metric label="Avg Degree" value="3.82" />
        </div>
      </div>

      {/* Bottom-left hint */}
      <div className="pointer-events-none absolute bottom-5 left-5 z-10 text-[10px] text-neutral-500 font-mono">
        move the cursor over the brain
      </div>

      {/* Selection close (placeholder for future) */}
      <button
        onClick={() => setSelected(null)}
        className="hidden"
        aria-hidden
      />
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-neutral-500">{label}</span>
      <span className="text-neutral-100 font-mono tabular-nums">{value}</span>
    </div>
  )
}
