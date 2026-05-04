import { NavLink } from "react-router-dom"
import { Network, FileText, FolderTree, Search, Settings, BarChart2, BookOpen, MessageSquare, Sparkles } from "lucide-react"
import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"
import { useTheme, type Theme } from "@/context/ThemeContext"

const THEMES: { id: Theme; color: string; label: string }[] = [
  { id: "teal", color: "#00c4bc", label: "Teal" },
  { id: "blue", color: "#3b82f6", label: "Blue" },
  { id: "purple", color: "#a855f7", label: "Purple" },
]

const links = [
  { to: "/graph",     label: "Graph",     icon: Network,       id: "01", desc: "Neural map" },
  { to: "/memories",  label: "Memories",  icon: FileText,      id: "02", desc: "Memory store" },
  { to: "/chat",      label: "Chat",      icon: MessageSquare, id: "05", desc: "Memory-augmented LLM" },
  { to: "/domains",   label: "Domains",   icon: FolderTree,    id: "03", desc: "Knowledge tree" },
  { to: "/retrieval", label: "Retrieval", icon: Search,        id: "04", desc: "Query engine" },
  { to: "/metrics",   label: "Metrics",   icon: BarChart2,     id: "06", desc: "Quality analytics" },
  { to: "/settings",  label: "Settings",  icon: Settings,      id: "07", desc: "Config" },
  { to: "/guide",     label: "Guide",     icon: BookOpen,      id: "08", desc: "How it works" },
  { to: "/about",     label: "About",     icon: Sparkles,      id: "09", desc: "Dedication" },
]

interface SidebarProps {
  onNavigate?: () => void
}

export function Sidebar({ onNavigate }: SidebarProps) {
  const [tick, setTick] = useState(0)
  const { theme, setTheme, colors } = useTheme()

  useEffect(() => {
    const t = setInterval(() => setTick(p => p + 1), 1000)
    return () => clearInterval(t)
  }, [])

  const hh = String(Math.floor(tick / 3600)).padStart(2, "0")
  const mm = String(Math.floor((tick % 3600) / 60)).padStart(2, "0")
  const ss = String(tick % 60).padStart(2, "0")

  return (
    <aside
      className="relative flex h-screen w-56 lg:w-56 flex-col font-mono overflow-hidden select-none"
      style={{
        background: "linear-gradient(180deg, #020d0d 0%, #010f0f 50%, #020d0d 100%)",
        borderRight: `1px solid ${colors.primaryBorder}`,
      }}
    >
      {/* Animated right-edge glow */}
      <div
        className="pointer-events-none absolute right-0 top-0 h-full w-px"
        style={{
          background: `linear-gradient(180deg, transparent 0%, ${colors.primaryDim} 25%, rgba(255,140,38,0.45) 55%, rgba(255,77,109,0.3) 82%, transparent 100%)`,
          boxShadow: `0 0 8px ${colors.primaryDim}`,
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

      {/* Logo */}
      <div
        className="relative px-5 pt-6 pb-4"
        style={{ borderBottom: "1px solid rgba(0, 196, 188,0.12)" }}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="relative shrink-0 h-14 w-14 flex items-center justify-center">
            <img
              src="/logo.png"
              alt="LocMemory logo"
              className="h-14 w-14 object-contain"
              style={{ filter: "drop-shadow(0 0 8px rgba(0, 196, 188, 0.6))", mixBlendMode: "screen" }}
            />
          </div>

          <div className="pt-1">
            <div
              className="text-[18px] font-bold tracking-wide"
              style={{ 
                color: colors.primaryText,
                textShadow: `0 0 12px ${colors.primaryGlow}`,
              }}
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

        {/* Theme switcher */}
        <div className="flex items-center gap-1.5 mt-3">
          <span className="text-[8px] uppercase tracking-[0.15em] text-neutral-600">Theme</span>
          <div className="flex gap-1">
            {THEMES.map((t) => (
              <button
                key={t.id}
                onClick={() => setTheme(t.id)}
                className={cn(
                  "w-4 h-4 rounded-full transition-all duration-200",
                  theme === t.id ? "ring-1 ring-offset-1 ring-offset-black" : "opacity-50 hover:opacity-80"
                )}
                style={{
                  background: t.color,
                  boxShadow: theme === t.id ? `0 0 8px ${t.color}` : `0 0 4px ${t.color}50`,
                  ringColor: t.color,
                }}
                title={t.label}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Nav */}
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
            onClick={onNavigate}
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
                    style={{ background: "linear-gradient(90deg, rgba(0, 196, 188,0.07), transparent)" }}
                  />
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

      {/* spacer */}
      <div className="flex-1" />

      {/* Footer */}
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