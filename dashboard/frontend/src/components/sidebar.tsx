import { NavLink } from "react-router-dom"
import { Network, FileText, FolderTree, Search, Settings, BarChart2 } from "lucide-react"
import { useEffect, useState, useRef } from "react"
import { api, type Stats } from "@/lib/api"
import { cn } from "@/lib/utils"
import { AnimatedNumber } from "@/components/hud"
import { usePrivacy } from "@/lib/privacy"

const links = [
  { to: "/graph",     label: "Graph",     icon: Network,    id: "01", desc: "Neural map" },
  { to: "/memories",  label: "Memories",  icon: FileText,   id: "02", desc: "Memory store" },
  { to: "/domains",   label: "Domains",   icon: FolderTree, id: "03", desc: "Knowledge tree" },
  { to: "/retrieval", label: "Retrieval", icon: Search,     id: "04", desc: "Query engine" },
  { to: "/metrics",   label: "Metrics",    icon: BarChart2,  id: "05", desc: "Quality analytics" },
  { to: "/settings",  label: "Settings",  icon: Settings,   id: "06", desc: "Config" },
]

function NeuralPulse() {
  return (
    <div className="px-4 py-4">
      <div className="text-[8px] uppercase tracking-[0.25em] text-neutral-600 mb-3">
        // NEURAL ACTIVITY
      </div>
      <div className="relative h-10 w-full overflow-hidden rounded-sm"
        style={{ background: "rgba(0, 196, 188,0.04)", border: "1px solid rgba(0, 196, 188,0.1)" }}>
        <NeuralWaveform />
      </div>
      <div className="mt-2 flex items-center justify-between text-[8px] text-neutral-700 uppercase tracking-wider">
        <span>SIGNAL</span>
        <span className="text-emerald-600/60 tabular-nums">ACTIVE</span>
      </div>
    </div>
  )
}

function NeuralWaveform() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<number>(0)
  const offsetRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const draw = () => {
      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)

      offsetRef.current += 0.04

      ctx.beginPath()
      ctx.strokeStyle = "rgba(0, 196, 188, 0.75)"
      ctx.lineWidth = 1.5
      ctx.shadowColor = "rgba(0, 196, 188, 0.8)"
      ctx.shadowBlur = 4

      for (let x = 0; x <= w; x += 2) {
        const t = (x / w) * Math.PI * 6 + offsetRef.current
        const y = h / 2 + Math.sin(t) * 8 + Math.sin(t * 2.3 + 1) * 4
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()

      // dim secondary wave — orange
      ctx.beginPath()
      ctx.strokeStyle = "rgba(255, 140, 38, 0.35)"
      ctx.lineWidth = 1
      ctx.shadowBlur = 0
      for (let x = 0; x <= w; x += 2) {
        const t = (x / w) * Math.PI * 4 + offsetRef.current * 0.7 + 2
        const y = h / 2 + Math.sin(t) * 5 + Math.cos(t * 1.7) * 3
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()

      frameRef.current = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(frameRef.current)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      width={176}
      height={40}
      className="absolute inset-0 h-full w-full"
    />
  )
}

export function Sidebar() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [prevStats, setPrevStats] = useState<Stats | null>(null)
  const [flashing, setFlashing] = useState({ nodes: false, edges: false, domains: false })
  const [tick, setTick] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchStats = async () => {
    try {
      const data = await api.stats()
      setPrevStats(stats)
      setStats(data)
    } catch (e) {
      console.error("Failed to fetch stats", e)
    }
  }

  useEffect(() => {
    fetchStats()
    intervalRef.current = setInterval(fetchStats, 10000)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [])

  useEffect(() => {
    const t = setInterval(() => setTick(p => p + 1), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (prevStats && stats) {
      const n = stats.nodes !== prevStats.nodes
      const e = stats.edges !== prevStats.edges
      const d = Object.keys(stats.domain_counts).length !== Object.keys(prevStats.domain_counts).length
      if (n || e || d) {
        setFlashing({ nodes: n, edges: e, domains: d })
        setTimeout(() => setFlashing({ nodes: false, edges: false, domains: false }), 600)
      }
    }
  }, [stats, prevStats])

  const domainCount = stats?.domain_counts ? Object.keys(stats.domain_counts).length : 0

  const hh = String(Math.floor(tick / 3600)).padStart(2, "0")
  const mm = String(Math.floor((tick % 3600) / 60)).padStart(2, "0")
  const ss = String(tick % 60).padStart(2, "0")

  return (
    <aside
      className="relative flex h-screen w-56 flex-col font-mono overflow-hidden select-none"
      style={{
        background: "linear-gradient(180deg, #020d0d 0%, #010f0f 50%, #020d0d 100%)",
        borderRight: "1px solid rgba(0, 196, 188, 0.12)",
      }}
    >
      {/* Animated right-edge glow */}
      <div
        className="pointer-events-none absolute right-0 top-0 h-full w-px"
        style={{
          background: "linear-gradient(180deg, transparent 0%, rgba(0, 196, 188,0.45) 25%, rgba(255,140,38,0.45) 55%, rgba(255,77,109,0.3) 82%, transparent 100%)",
          boxShadow: "0 0 8px rgba(0, 196, 188, 0.3)",
        }}
      />

      {/* Top ambient glow */}
      <div
        className="pointer-events-none absolute top-0 left-0 w-full h-32"
        style={{
          background: "radial-gradient(ellipse at 30% 0%, rgba(0, 196, 188,0.08) 0%, transparent 70%)",
        }}
      />

      {/* Bottom ambient glow */}
      <div
        className="pointer-events-none absolute bottom-0 left-0 w-full h-32"
        style={{
          background: "radial-gradient(ellipse at 30% 100%, rgba(200,168,255,0.06) 0%, transparent 70%)",
        }}
      />

      {/* Corner brackets */}
      <div className="pointer-events-none absolute top-2 left-2 h-5 w-5 border-t border-l border-emerald-400/50"
        style={{ filter: "drop-shadow(0 0 3px rgba(0, 196, 188,0.5))" }} />
      <div className="pointer-events-none absolute bottom-2 left-2 h-5 w-5 border-b border-l"
        style={{ borderColor: "rgba(255,140,38,0.45)", filter: "drop-shadow(0 0 3px rgba(255,140,38,0.4))" }} />
      <div className="pointer-events-none absolute top-2 right-2 h-5 w-5 border-t border-r"
        style={{ borderColor: "rgba(255,77,109,0.35)", filter: "drop-shadow(0 0 3px rgba(255,77,109,0.3))" }} />

      {/* ── Logo ── */}
      <div
        className="relative px-5 pt-6 pb-4"
        style={{ borderBottom: "1px solid rgba(0, 196, 188,0.12)" }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="relative shrink-0 h-10 w-10 flex items-center justify-center">
            <img
              src="/logo.png"
              alt="LocMemory logo"
              className="h-10 w-10 object-contain"
              style={{ filter: "drop-shadow(0 0 6px rgba(0, 196, 188, 0.6))", mixBlendMode: "screen" }}
            />
          </div>

          <div>
            <div
              className="text-[15px] font-bold tracking-wide text-emerald-300"
              style={{ textShadow: "0 0 12px rgba(0, 196, 188,0.6)" }}
            >
              LocMemory
            </div>
            <div className="text-[8px] text-emerald-700/80 uppercase tracking-[0.25em]">
              v0.1.0
            </div>
          </div>
        </div>

        {/* Status + uptime row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span
              className="h-1.5 w-1.5 rounded-full bg-green-400"
              style={{ boxShadow: "0 0 6px rgba(74,222,128,0.9)", animation: "pulse 2s ease-in-out infinite" }}
            />
            <span className="text-[8px] uppercase tracking-[0.2em] text-green-400/80">ONLINE</span>
          </div>
          <span
            className="text-[9px] tabular-nums text-emerald-700/60"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {hh}:{mm}:{ss}
          </span>
        </div>
      </div>

      {/* ── Nav ── */}
      <nav className="px-3 py-4 space-y-1">
        <div className="px-2 mb-3 flex items-center gap-2">
          <div className="h-px flex-1" style={{ background: "linear-gradient(to right, rgba(0, 196, 188,0.3), transparent)" }} />
          <span className="text-[8px] uppercase tracking-[0.25em] text-emerald-700/60">NAV</span>
          <div className="h-px flex-1" style={{ background: "linear-gradient(to left, rgba(0, 196, 188,0.3), transparent)" }} />
        </div>

        {links.map(({ to, label, icon: Icon, id, desc }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "group relative flex items-center gap-2.5 px-3 py-2.5 rounded-sm text-[11px] uppercase tracking-widest transition-all duration-200 overflow-hidden",
                isActive
                  ? "text-emerald-300"
                  : "text-neutral-500 hover:text-emerald-100"
              )
            }
            style={({ isActive }) => isActive ? {
              background: "linear-gradient(90deg, rgba(0, 196, 188,0.12) 0%, rgba(0, 196, 188,0.04) 100%)",
              boxShadow: "inset 0 0 20px rgba(0, 196, 188,0.05)",
            } : {}}
          >
            {({ isActive }) => (
              <>
                {/* Hover bg */}
                {!isActive && (
                  <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                    style={{ background: "linear-gradient(90deg, rgba(0, 196, 188,0.07), transparent)" }} />
                )}

                {/* Active left bar */}
                <div
                  className="absolute left-0 top-1 bottom-1 w-0.5 rounded-full transition-all duration-300"
                  style={{
                    background: isActive ? "linear-gradient(180deg, transparent, #00c4bc, transparent)" : "transparent",
                    boxShadow: isActive ? "0 0 8px rgba(0, 196, 188,0.8)" : "none",
                    opacity: isActive ? 1 : 0,
                  }}
                />

                {/* ID */}
                <span className={cn("text-[8px] w-4 tabular-nums shrink-0", isActive ? "text-emerald-500/50" : "text-neutral-700 group-hover:text-neutral-600")}>
                  {id}
                </span>

                {/* Icon */}
                <Icon
                  className="w-3.5 h-3.5 shrink-0 transition-all duration-200"
                  style={isActive
                    ? { filter: "drop-shadow(0 0 5px rgba(0, 196, 188,0.8))", color: "#00c4bc" }
                    : {}}
                />

                {/* Label + desc */}
                <div className="flex flex-col min-w-0">
                  <span className="leading-none">{label}</span>
                  {isActive && (
                    <span className="text-[7px] text-emerald-600/60 normal-case tracking-wider mt-0.5 leading-none">
                      {desc}
                    </span>
                  )}
                </div>

                {/* Active indicator */}
                {isActive && (
                  <span
                    className="ml-auto h-1 w-1 rounded-full shrink-0 bg-emerald-400"
                    style={{ boxShadow: "0 0 6px rgba(0, 196, 188,0.9)", animation: "pulse 2s ease-in-out infinite" }}
                  />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* ── Neural activity waveform ── */}
      <div className="flex-1 flex flex-col justify-center">
        <NeuralPulse />
      </div>

      {/* ── Stats ── */}
      <div
        className="px-4 py-3"
        style={{ borderTop: "1px solid rgba(0, 196, 188,0.1)" }}
      >
        <div className="flex items-center gap-2 mb-3">
          <div className="h-px flex-1" style={{ background: "linear-gradient(to right, rgba(0, 196, 188,0.3), transparent)" }} />
          <span className="text-[8px] uppercase tracking-[0.25em] text-emerald-700/60">METRICS</span>
          <div className="h-px flex-1" style={{ background: "linear-gradient(to left, rgba(0, 196, 188,0.3), transparent)" }} />
        </div>

        <div className="space-y-1.5">
          {[
            { label: "NODES",   value: stats?.nodes,           flash: flashing.nodes,   color: "#00c4bc" },
            { label: "EDGES",   value: stats?.edges,           flash: flashing.edges,   color: "#ff8c26" },
            { label: "DOMAINS", value: domainCount || null,    flash: flashing.domains, color: "#ffd700" },
          ].map(({ label, value, flash, color }) => (
            <div
              key={label}
              className={cn(
                "relative flex items-center justify-between px-2 py-1.5 rounded-sm transition-all duration-300"
              )}
              style={{ background: flash ? `rgba(0, 196, 188, 0.05)` : undefined }}
            >
              {/* color accent */}
              <div className="absolute left-0 top-1/2 -translate-y-1/2 h-3 w-0.5 rounded-full"
                style={{ background: color, boxShadow: `0 0 5px ${color}` }} />
              <span className="pl-2 text-[9px] uppercase tracking-wider text-neutral-600">{label}</span>
              <span
                className="text-[12px] font-semibold tabular-nums"
                style={{
                  color: flash ? "#00c4bc" : color,
                  textShadow: flash ? `0 0 10px rgba(0, 196, 188,0.9)` : `0 0 8px ${color}80`,
                }}
              >
                {value != null
                  ? typeof value === "number"
                    ? <AnimatedNumber value={value} duration={500} />
                    : value
                  : <span className="text-neutral-700 text-[10px]">—</span>}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Footer ── */}
      <div
        className="px-5 py-2.5 flex items-center justify-between"
        style={{ borderTop: "1px solid rgba(0, 196, 188,0.08)" }}
      >
        <span className="text-[8px] uppercase tracking-[0.2em] text-neutral-700">
          BUILD 2025.04
        </span>
        <div className="flex gap-1">
          {[0, 1, 2].map(i => (
            <div
              key={i}
              className="h-1 w-1 rounded-full bg-emerald-900"
              style={{
                animation: `pulse ${1.2 + i * 0.3}s ease-in-out infinite`,
                animationDelay: `${i * 0.2}s`,
              }}
            />
          ))}
        </div>
      </div>

    </aside>
  )
}
