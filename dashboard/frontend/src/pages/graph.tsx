import { useState, useEffect } from "react"
import { BrainScene } from "@/components/BrainScene"
import { MatrixRain } from "@/components/MatrixRain"
import {
  AnimatedNumber,
  StatusDot,
  HudBracket,
  ScanlineOverlay,
} from "@/components/hud"
import { api, type Stats } from "@/lib/api"
import { RetrievalConsole } from "@/components/RetrievalConsole"
import { MemoryInspector } from "@/components/MemoryInspector"
import { HebbianPanel } from "@/components/HebbianPanel"
import { PatternsPanel } from "@/components/PatternsPanel"
import { DomainsPanel } from "@/components/DomainsPanel"
import { useTheme } from "@/lib/theme"

interface RLStatus {
  enabled: boolean
  available: boolean
  model_path?: string
  candidate_pool_size?: number
  top_k?: number
  token_budget?: number
  message?: string
}

const TIER_COLORS = ["#00ff88", "#00e5ff", "#aaff00", "#00ff66"] as const
const TIER_RGB = ["0,255,136", "0,229,255", "170,255,0", "0,255,102"] as const

const TIER_LABELS = [
  "Context",
  "Anchor",
  "Leaf",
  "Procedural",
]
const TIER_SUB = ["T1", "T2", "T3", "T4"]
const TIER_DESC = [
  "Recent session facts — high recency, fast decay",
  "Consolidated summaries via Louvain clustering",
  "Stable long-term facts from conversations",
  "Cross-domain behavioural patterns",
]

export function GraphPage() {
  const { theme } = useTheme()
  const [selected, setSelected] = useState<string | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [rlStatus, setRlStatus] = useState<RLStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [uptime, setUptime] = useState(0)

  useEffect(() => {
    api.stats()
      .then((data) => {
        setStats(data)
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })

    api.rlStatus()
      .then((data) => setRlStatus(data))
      .catch(() => {})
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
        Number(stats.tier_counts["1"] ?? stats.tier_counts["context"] ?? 0),
        Number(stats.tier_counts["2"] ?? stats.tier_counts["anchor"] ?? 0),
        Number(stats.tier_counts["3"] ?? stats.tier_counts["leaf"] ?? 0),
        Number(stats.tier_counts["4"] ?? stats.tier_counts["procedural"] ?? 0),
      ]
    : [0, 0, 0, 0]

  const totalNodes = stats?.nodes ?? 0
  const totalEdges = stats?.edges ?? 0

  return (
    <div className={`relative h-full w-full overflow-hidden font-mono ${theme === "dark" ? "bg-[#020d0d]" : "bg-slate-100"}`}>
      {/* Layer 0: Matrix rain background - dark only */}
      {theme === "dark" && (
        <div className="absolute inset-0 z-0">
          <MatrixRain
            fontSize={14}
            speed={60}
            foreground="#00ff88"
            background="#020d0d"
            opacity={0.07}
          />
        </div>
      )}

      {/* Layer 1: Dark base */}
      <div className={`absolute inset-0 z-[1] ${theme === "dark" ? "bg-[#020d0d]/95" : "bg-slate-100/95"}`} />

      {/* Layer 2: Radial glow */}
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

      {/* CSS Grid Layout: [center 1fr] [right 280px] */}
      <div
        className="relative z-10 h-full w-full"
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 280px",
          gridTemplateRows: "auto 1fr auto",
          gap: "8px",
          padding: "8px",
        }}
      >
        {/* ── Top Status Bar (spans both columns) ── */}
        <div
          className="col-span-2 flex items-center justify-center gap-3 py-2 px-4"
          style={{
            background: "rgba(0,20,15,0.85)",
            border: "1px solid rgba(0,255,180,0.15)",
            borderRadius: "8px",
          }}
        >
          <div className="text-[10px] uppercase tracking-[0.25em] text-emerald-500/60 flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              SYS
            </span>
            <span className="text-emerald-500/30">●</span>
            <span>NODES: {loading ? "…" : totalNodes}</span>
            <span className="text-emerald-500/30">●</span>
            <span>EDGES: {loading ? "…" : totalEdges}</span>
            <span className="text-emerald-500/30">●</span>
            <span>UPTIME: {formatUptime(uptime)}</span>
            <span className="text-emerald-500/30">●</span>
            <span className={rlStatus?.available ? "text-purple-400" : "text-neutral-600"}>
              RL: {rlStatus?.available ? "ACTIVE" : rlStatus?.enabled ? "LOADING" : "OFF"}
            </span>
            <span className="text-emerald-500/30">●</span>
            <span>STATUS: NOMINAL</span>
          </div>
        </div>

        {/* ── Center Column: Edge Dynamics + Graph + Retrieval Console ── */}
        <div
          className="row-span-2 flex flex-col gap-2"
          style={{
            display: "grid",
            gridTemplateRows: "1fr auto",
            gap: "8px",
          }}
        >
          {/* Top: Edge Dynamics + Cognitive Ops (left) + Graph (right) */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "270px 1fr",
              gap: "8px",
            }}
          >
            {/* Left: Edge Dynamics + Cognitive Ops */}
            <div className="flex flex-col gap-2 overflow-hidden">
              <HebbianPanel />
              <PatternsPanel />
            </div>

            {/* Right: 3D Graph Canvas */}
            <div
              className="relative overflow-hidden"
              style={{
                background: "rgba(0,20,15,0.85)",
                border: "1px solid rgba(0,255,180,0.15)",
                borderRadius: "8px",
              }}
            >
              <BrainScene
                className="absolute inset-0"
                selectedId={selected}
                onNodeSelect={setSelected}
                showEdges={true}
              />
            </div>
          </div>

          {/* Bottom: Retrieval Console (spans full width of center column) */}
          <div style={{ minHeight: "180px" }}>
            <RetrievalConsole onSelect={setSelected} />
          </div>
        </div>

        {/* ── Right Column: Memory Tiers + Knowledge Domains (50/50) ── */}
        <div
          className="row-span-2 flex flex-col gap-2"
          style={{
            background: "rgba(0,20,15,0.85)",
            border: "1px solid rgba(0,255,180,0.15)",
            borderRadius: "8px",
            padding: "16px",
          }}
        >
          {/* SYS.MEM.01 - Memory Tiers (top) */}
          <div className="flex-1 flex flex-col min-h-0">
            <div
              className="flex items-center justify-between pb-2 mb-2"
              style={{ borderBottom: "2px solid #00ff88" }}
            >
              <span className="text-[10px] uppercase tracking-widest text-neutral-400">
                Memory Tiers
              </span>
              <StatusDot label="ACTIVE" color="#00ff88" />
            </div>
            <div className="flex-1 space-y-2 overflow-y-auto">
              {TIER_LABELS.map((label, i) => {
                const count = tierCounts[i]
                const pct = totalNodes > 0 ? Math.round((count / totalNodes) * 100) : 0
                return (
                  <div key={i} className="px-2 py-1.5">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className="inline-block h-2 w-2 shrink-0 rounded-full"
                        style={{ backgroundColor: TIER_COLORS[i], boxShadow: `0 0 6px ${TIER_COLORS[i]}` }}
                      />
                      <span className="text-[9px] text-neutral-500 font-mono">{TIER_SUB[i]}</span>
                      <span className="text-[11px] text-neutral-100 flex-1">{label}</span>
                      <span className="text-[11px] font-mono tabular-nums" style={{ color: TIER_COLORS[i] }}>
                        {loading ? "…" : <AnimatedNumber value={count} duration={1200} className="tabular-nums" />}
                      </span>
                      {!loading && totalNodes > 0 && (
                        <span className="text-[9px] text-neutral-600 w-8 text-right">{pct}%</span>
                      )}
                    </div>
                    <div className="h-0.5 w-full rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.05)" }}>
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{ width: loading ? "0%" : `${pct}%`, background: TIER_COLORS[i], boxShadow: `0 0 4px ${TIER_COLORS[i]}80` }}
                      />
                    </div>
                    <div className="text-[9px] text-neutral-600 mt-0.5 leading-snug">{TIER_DESC[i]}</div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* SYS.DOM.04 - Knowledge Domains (bottom) */}
          <div className="flex-1 min-h-0 mt-4 overflow-y-auto">
            <DomainsPanel />
          </div>
        </div>
      </div>

      {/* Layer 20: Vignette */}
      <div
        className="pointer-events-none absolute inset-0 z-20"
        style={{
          background: "radial-gradient(ellipse at center, transparent 50%, rgba(0,5,16,0.85) 100%)",
        }}
      />

      <ScanlineOverlay />

      <HudBracket position="tl" size={48} />
      <HudBracket position="tr" size={48} />
      <HudBracket position="bl" size={48} />
      <HudBracket position="br" size={48} />

      {/* Memory inspector */}
      <MemoryInspector
        memoryId={selected}
        onClose={() => setSelected(null)}
        onDeleted={() => setSelected(null)}
      />
    </div>
  )
}