import { useState, useEffect } from "react"
import { BrainScene } from "@/components/BrainScene"
import { MatrixRain } from "@/components/MatrixRain"
import {
  AnimatedNumber,
  StatusDot,
  HudBracket,
  ScanlineOverlay,
  DataFeedLine,
  HudPanel,
  BottomStatusBar,
} from "@/components/hud"
import { api, type Stats } from "@/lib/api"
import { RetrievalConsole } from "@/components/RetrievalConsole"
import { MemoryInspector } from "@/components/MemoryInspector"

const TIER_COLORS = ["#3b82f6", "#06b6d4", "#9ec5e8", "#a855f7"] as const
const TIER_RGB = ["59,130,246", "6,182,212", "158,197,232", "168,85,247"] as const
const TIER_LABELS = [
  "Core Context",
  "Anchor Memories",
  "Leaf Memories",
  "Procedural Memories",
]
const TIER_SUB = ["(Deep)", "(Mid)", "(Facts)", ""]
const TIER_DESC = [
  "High-levle semantic hubs",
  "Stable reference points",
  "Atomic facts & observations",
  "Skills, processes & workflows",
]

const DEFAULT_TIER_COUNTS = [1435, 875, 770, 420]
const TOTAL_NODES = DEFAULT_TIER_COUNTS.reduce((a, b) => a + b, 0)
const DEFAULT_TOTAL_EDGES = 6681

const EVENT_TYPES = ["NODE", "EDGE", "CLUSTER", "RETRIEVAL", "HEBB"] as const
type EventType = typeof EVENT_TYPES[number]

interface LogEvent {
  id: number
  timestamp: string
  type: EventType
  message: string
}

function formatTimestamp(date: Date): string {
  const hh = String(date.getHours()).padStart(2, "0")
  const mm = String(date.getMinutes()).padStart(2, "0")
  const ss = String(date.getSeconds()).padStart(2, "0")
  const ms = String(date.getMilliseconds()).padStart(3, "0")
  return `${hh}:${mm}:${ss}.${ms}`
}

function generateEvent(): LogEvent {
  const type = EVENT_TYPES[Math.floor(Math.random() * EVENT_TYPES.length)]
  const messages: Record<EventType, string[]> = {
    NODE: ["+1 [anchor.memories]", "+1 [leaf.facts]", "+1 [core.context]", "+1 [procedural]"],
    EDGE: ["STRENGTHENED 0.84→0.91", "CREATED 0.67", "WEAKENED 0.32→0.28", "REINFORCED"],
    CLUSTER: ["DETECTED #47", "MERGED #12→#13", "EXPANDED #08", "REBALANCED"],
    RETRIEVAL: ["QUERY processed 142ms", "QUERY processed 89ms", "QUERY processed 201ms", "QUERY processed 67ms"],
    HEBB: ["UPDATE 23 edges", "UPDATE 7 edges", "UPDATE 15 edges", "CONSOLIDATION merged 3 nodes"],
  }
  const msg = messages[type][Math.floor(Math.random() * messages[type].length)]
  return {
    id: Date.now() + Math.random(),
    timestamp: formatTimestamp(new Date()),
    type,
    message: msg,
  }
}

export function GraphPage() {
  const [selected, setSelected] = useState<string | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [loading, setLoading] = useState(true)
  const [reinitting, setReinitting] = useState(false)
  const [uptime, setUptime] = useState(0)
  const [logs, setLogs] = useState<LogEvent[]>([])

  useEffect(() => {
    api.stats()
      .then((data) => {
        setStats(data)
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      setUptime((prev) => prev + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      const newEvent = generateEvent()
      setLogs((prev) => [newEvent, ...prev].slice(0, 8))
    }, 3000)
    setLogs([generateEvent()])
    return () => clearInterval(interval)
  }, [])

  const handleReinit = () => {
    setReinitting(true)
    setTimeout(() => setReinitting(false), 1500)
  }

  const formatUptime = (sec: number) => {
    const h = Math.floor(sec / 3600)
    const m = Math.floor((sec % 3600) / 60)
    const s = sec % 60
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
  }

  const tierCounts = stats?.tier_counts
    ? [
        stats.tier_counts["Core Context"] || 0,
        stats.tier_counts["Anchor Memories"] || 0,
        stats.tier_counts["Leaf Memories"] || 0,
        stats.tier_counts["Procedural Memories"] || 0,
      ]
    : DEFAULT_TIER_COUNTS

  const totalNodes = stats?.nodes || TOTAL_NODES
  const totalEdges = stats?.edges || DEFAULT_TOTAL_EDGES

  const panelMetrics = [
    { label: "Total Nodes", value: totalNodes },
    { label: "Total Edges", value: totalEdges },
    { label: "Communities", value: null as number | null },
    { label: "Density", value: "1.09e-3" },
    { label: "Avg Degree", value: "3.82" },
  ]

  const getEventColor = (type: EventType) => {
    switch (type) {
      case "NODE":
        return "text-blue-400"
      case "EDGE":
        return "text-cyan-400"
      case "CLUSTER":
        return "text-purple-400"
      case "RETRIEVAL":
        return "text-neutral-400"
      case "HEBB":
        return "text-cyan-400"
      default:
        return "text-neutral-400"
    }
  }

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#000510] font-mono">
      {/* Layer 0: MatrixRain background */}
      <div className="absolute inset-0 z-0">
        <MatrixRain
          fontSize={14}
          speed={60}
          foreground="#06b6d4"
          background="#000510"
          opacity={0.1}
        />
      </div>

      {/* Layer 1: Solid dark background for contrast */}
      <div className="absolute inset-0 z-[1] bg-[#000510]/95" />

      {/* Layer 2: Subtle gradient overlay */}
      <div
        className="pointer-events-none absolute inset-0 z-[2]"
        style={{
          background: `
            radial-gradient(circle at 0% 0%, rgba(59,130,246,0.12), transparent 35%),
            radial-gradient(circle at 100% 0%, rgba(6,182,212,0.12), transparent 35%),
            radial-gradient(circle at 0% 100%, rgba(59,130,246,0.10), transparent 35%),
            radial-gradient(circle at 100% 100%, rgba(6,182,212,0.12), transparent 35%)
          `,
        }}
      />

      {/* Layer 10: Brain visualization - must be clearly visible */}
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
            "radial-gradient(ellipse at center, transparent 50%, rgba(0,5,16,0.8) 100%)",
        }}
      />

      <ScanlineOverlay />

      <HudBracket position="tl" size={48} />
      <HudBracket position="tr" size={48} />
      <HudBracket position="bl" size={48} />
      <HudBracket position="br" size={48} />

      <DataFeedLine startX={window.innerWidth - 292} startY={110} endX={window.innerWidth - 360} endY={200} delay={900} />
      <DataFeedLine startX={window.innerWidth - 292} startY={460} endX={window.innerWidth - 350} endY={400} delay={1100} />

      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20">
        <div className="text-[10px] uppercase tracking-[0.25em] text-cyan-500/60 flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            SYS
          </span>
          <span className="separator-pulse">●</span>
          <span>NODES: {loading ? "..." : totalNodes}</span>
          <span className="separator-pulse">●</span>
          <span>EDGES: {loading ? "..." : totalEdges}</span>
          <span className="separator-pulse">●</span>
          <span>UPTIME: {formatUptime(uptime)}</span>
          <span className="separator-pulse">●</span>
          <span>STATUS: NOMINAL</span>
        </div>
      </div>

      <div className="absolute left-5 top-1/2 -translate-y-1/2 z-10 w-64">
        <HudPanel id="SYS.LOG.03" className="hud-panel" progressValue={45}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] uppercase tracking-widest text-neutral-400">
              Neural Activity Log
            </div>
            <StatusDot label="LIVE" color="#22d3ee" />
          </div>
          <div className="space-y-0.5 max-h-[240px] overflow-hidden">
            {logs.map((log) => (
              <div
                key={log.id}
                className="log-entry flex items-start gap-2 py-1 px-1 text-[10px] border-l-2 border-transparent hover:bg-white/5 transition-colors"
              >
                <span className="text-neutral-600 shrink-0 font-mono">{log.timestamp}</span>
                <span className={`shrink-0 font-medium ${getEventColor(log.type)}`}>
                  {log.type}
                </span>
                <span className="text-neutral-300 truncate">{log.message}</span>
              </div>
            ))}
          </div>
        </HudPanel>
      </div>

      <div className="absolute top-5 right-5 z-10 w-72 p-4">
        <HudPanel id="SYS.MEM.01" className="hud-panel" progressValue={78}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] uppercase tracking-widest text-neutral-400">
              Memory Tiers
            </div>
            <StatusDot label="ACTIVE" color="#22d3ee" />
          </div>
          <div className="space-y-1.5">
            {TIER_LABELS.map((label, i) => (
              <div
                key={i}
                className="tier-row flex items-start gap-2.5 px-2 py-1.5"
                style={{
                  "--tier-rgb": TIER_RGB[i],
                } as React.CSSProperties}
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
                    <span className="font-400 text-neutral-500">
                      {TIER_SUB[i]}
                    </span>
                  </div>
                  <div className="text-[10px] text-neutral-500 leading-snug">
                    {TIER_DESC[i]}
                  </div>
                </div>
                <div className="text-[11px] font-500 tabular-nums text-neutral-100">
                  <AnimatedNumber
                    value={tierCounts[i]}
                    duration={1200}
                    className="tabular-nums"
                  />
                </div>
              </div>
            ))}
          </div>
        </HudPanel>
      </div>

      <div className="absolute top-[340px] right-5 z-10 w-72 p-4">
        <HudPanel id="SYS.GRP.02" className="hud-panel hud-panel-2" progressValue={92}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] uppercase tracking-widest text-neutral-400">
              Graph Metrics
            </div>
            <StatusDot label="STABLE" color="#3b82f6" />
          </div>
          <div className="space-y-2">
            {panelMetrics.map((m, i) => (
              <div
                key={i}
                className="flex items-center justify-between text-[12px]"
              >
                <span className="text-neutral-500">{m.label}</span>
                <span className="font-500 tabular-nums text-neutral-100">
                  {m.value === null ? (
                    "—"
                  ) : typeof m.value === "number" ? (
                    <AnimatedNumber value={m.value} duration={1200} />
                  ) : (
                    m.value
                  )}
                </span>
              </div>
            ))}
          </div>
        </HudPanel>
      </div>

      <div className="pointer-events-none absolute bottom-8 left-5 z-10 text-[10px] text-neutral-500 font-400">
        move the cursor over the brain
      </div>

      <RetrievalConsole onSelect={setSelected} />

      <MemoryInspector
        memoryId={selected}
        onClose={() => setSelected(null)}
        onDeleted={() => setSelected(null)}
      />

      <BottomStatusBar />

      <button
        onClick={handleReinit}
        className="absolute right-5 bottom-10 z-30 cursor-pointer border bg-transparent px-3 py-1.5 text-[10px] font-mono transition-all duration-200 reinit-button"
        style={{
          borderColor: reinitting ? "rgba(34,211,238,0.8)" : "rgba(34,211,238,0.4)",
          color: reinitting ? "rgba(34,211,238,0.9)" : "rgba(34,211,238,0.6)",
        }}
        aria-label="Toggle boot sequence"
      >
        {reinitting ? "[ REINITIALIZING... ]" : "[ ▋ REINIT ]"}
      </button>

    </div>
  )
}