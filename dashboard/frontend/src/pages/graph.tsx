import { useState, useEffect } from "react"
import { BrainScene } from "@/components/BrainScene"
import { MatrixRain } from "@/components/MatrixRain"
import {
  AnimatedNumber,
  StatusDot,
  HudBracket,
  ScanlineOverlay,
  HudPanel,
  BottomStatusBar,
} from "@/components/hud"
import { api, type Stats } from "@/lib/api"
import { RetrievalConsole } from "@/components/RetrievalConsole"
import { MemoryInspector } from "@/components/MemoryInspector"
import { MemoryCreator } from "@/components/MemoryCreator"
import { HebbianPanel } from "@/components/HebbianPanel"
import { PatternsPanel } from "@/components/PatternsPanel"
import { DomainsPanel } from "@/components/DomainsPanel"

// Bioluminescent tier palette — matches BrainScene glow colors
const TIER_COLORS = ["#00ff88", "#00e5ff", "#aaff00", "#00ff66"] as const
const TIER_RGB    = ["0,255,136", "0,229,255", "170,255,0", "0,255,102"] as const

const TIER_LABELS = [
  "Core Context",
  "Anchor Memories",
  "Leaf Memories",
  "Procedural Memories",
]
const TIER_SUB = ["(Deep)", "(Mid)", "(Facts)", "(Skills)"]
const TIER_DESC = [
  "High-level semantic hubs",
  "Stable reference points",
  "Atomic facts & observations",
  "Skills, processes & workflows",
]

export function GraphPage() {
  const [selected, setSelected] = useState<string | null>(null)
  const [stats, setStats]       = useState<Stats | null>(null)
  const [loading, setLoading]   = useState(true)
  const [uptime, setUptime]     = useState(0)

  useEffect(() => {
    api.stats()
      .then((data) => { setStats(data); setLoading(false) })
      .catch(() => { setLoading(false) })
  }, [])

  useEffect(() => {
    const id = setInterval(() => setUptime((p) => p + 1), 1000)
    return () => clearInterval(id)
  }, [])

  const formatUptime = (sec: number) => {
    const h = Math.floor(sec / 3600)
    const m = Math.floor((sec % 3600) / 60)
    const s = sec % 60
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
  }

  const tierCounts = stats?.tier_counts
    ? [
        stats.tier_counts["context"]    || 0,
        stats.tier_counts["anchor"]     || 0,
        stats.tier_counts["leaf"]       || 0,
        stats.tier_counts["procedural"] || 0,
      ]
    : [0, 0, 0, 0]

  const totalNodes = stats?.nodes ?? 0
  const totalEdges = stats?.edges ?? 0

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#020d0d] font-mono">
      {/* Layer 0: Matrix rain background */}
      <div className="absolute inset-0 z-0">
        <MatrixRain
          fontSize={14}
          speed={60}
          foreground="#00ff88"
          background="#020d0d"
          opacity={0.07}
        />
      </div>

      {/* Layer 1: Dark base for contrast */}
      <div className="absolute inset-0 z-[1] bg-[#020d0d]/95" />

      {/* Layer 2: Subtle radial glow corners */}
      <div
        className="pointer-events-none absolute inset-0 z-[2]"
        style={{
          background: `
            radial-gradient(circle at 0% 0%,   rgba(0,255,136,0.06), transparent 35%),
            radial-gradient(circle at 100% 0%,  rgba(0,229,255,0.06), transparent 35%),
            radial-gradient(circle at 0% 100%,  rgba(0,255,136,0.05), transparent 35%),
            radial-gradient(circle at 100% 100%, rgba(0,229,255,0.06), transparent 35%)
          `,
        }}
      />

      {/* Layer 10: Brain visualization */}
      <div className="absolute inset-0 z-10">
        <BrainScene
          className="absolute inset-0"
          selectedId={selected}
          onNodeSelect={setSelected}
          showEdges={true}
        />
      </div>

      {/* Layer 20: Vignette */}
      <div
        className="pointer-events-none absolute inset-0 z-20"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 50%, rgba(0,5,16,0.85) 100%)",
        }}
      />

      <ScanlineOverlay />

      <HudBracket position="tl" size={48} />
      <HudBracket position="tr" size={48} />
      <HudBracket position="bl" size={48} />
      <HudBracket position="br" size={48} />

      {/* ── Top status bar ── */}
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30">
        <div className="text-[10px] uppercase tracking-[0.25em] text-emerald-500/60 flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            SYS
          </span>
          <span className="separator-pulse">●</span>
          <span>NODES: {loading ? "…" : totalNodes}</span>
          <span className="separator-pulse">●</span>
          <span>EDGES: {loading ? "…" : totalEdges}</span>
          <span className="separator-pulse">●</span>
          <span>UPTIME: {formatUptime(uptime)}</span>
          <span className="separator-pulse">●</span>
          <span>STATUS: NOMINAL</span>
        </div>
      </div>

      {/* ── Left panel: Hebbian edge dynamics ── */}
      <div className="hidden md:block absolute left-5 top-8 z-30 w-64">
        <HebbianPanel />
      </div>

      {/* ── Right panel: Memory Tiers ── */}
      <div className="hidden md:block absolute top-5 right-5 z-30 w-72 p-4">
        <HudPanel id="SYS.MEM.01" className="hud-panel" progressValue={78}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] uppercase tracking-widest text-neutral-400">
              Memory Tiers
            </div>
            <StatusDot label="ACTIVE" color="#00ff88" />
          </div>
          <div className="space-y-1.5">
            {TIER_LABELS.map((label, i) => (
              <div
                key={i}
                className="flex items-start gap-2.5 px-2 py-1.5"
                style={{ "--tier-rgb": TIER_RGB[i] } as React.CSSProperties}
              >
                <span
                  className="mt-1.5 inline-block h-2.5 w-2.5 shrink-0 rounded-full hud-entity-dot"
                  style={{
                    backgroundColor: TIER_COLORS[i],
                    boxShadow: `0 0 8px ${TIER_COLORS[i]}`,
                    animationDelay: `${i * 0.15}s`,
                  }}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-[12px] font-500 text-neutral-100">
                    {label}{" "}
                    <span className="font-400 text-neutral-500">{TIER_SUB[i]}</span>
                  </div>
                  <div className="text-[10px] text-neutral-500 leading-snug">
                    {TIER_DESC[i]}
                  </div>
                </div>
                <div className="text-[11px] font-500 tabular-nums text-neutral-100">
                  {loading ? (
                    <span className="text-neutral-600">…</span>
                  ) : (
                    <AnimatedNumber value={tierCounts[i]} duration={1200} className="tabular-nums" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </HudPanel>
      </div>

      {/* ── Right panel: Knowledge Domains (real data) ── */}
      <div className="hidden md:block absolute top-[340px] right-5 z-30 w-72 p-4">
        <DomainsPanel />
      </div>

      {/* ── Right panel: Cognitive Ops (consolidate + detect patterns) ── */}
      <div className="hidden xl:block absolute top-[555px] right-5 z-30 w-72 p-4">
        <PatternsPanel />
      </div>

      {/* ── Bottom left: Memory creator + hint ── */}
      <div className="absolute bottom-8 left-5 z-30 flex flex-col items-start gap-2">
        <MemoryCreator
          onCreated={() => {
            api.stats().then((data) => setStats(data)).catch(() => {})
          }}
        />
        <div className="pointer-events-none text-[10px] text-neutral-500">
          click a node to inspect · drag to rotate
        </div>
      </div>

      {/* ── Retrieval console (bottom center) ── */}
      <RetrievalConsole onSelect={setSelected} />

      {/* ── Memory inspector (opens on node select) ── */}
      <MemoryInspector
        memoryId={selected}
        onClose={() => setSelected(null)}
        onDeleted={() => setSelected(null)}
      />

      <BottomStatusBar />
    </div>
  )
}
