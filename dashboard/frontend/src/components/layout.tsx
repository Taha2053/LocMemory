import { useState, useEffect } from "react"
import { Outlet, NavLink } from "react-router-dom"
import { Sidebar } from "./sidebar"
import { Menu, X } from "lucide-react"
import { useTheme } from "@/context/ThemeContext"

function FooterBar() {
  const [time, setTime] = useState(() => new Date())
  const { colors } = useTheme()

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const hh = String(time.getHours()).padStart(2, "0")
  const mm = String(time.getMinutes()).padStart(2, "0")
  const ss = String(time.getSeconds()).padStart(2, "0")

  return (
    <footer
      className="shrink-0 flex items-center justify-between px-5 font-mono select-none"
      style={{
        height: 30,
        borderTop: `1px solid ${colors.primaryBorder}`,
        background: "linear-gradient(90deg, #020d0d 0%, #010f0f 50%, #020d0d 100%)",
        boxShadow: `0 -1px 20px ${colors.primaryDim}`,
      }}
    >
      {/* Left — system tag */}
      <div className="flex items-center gap-2">
        <div
          className="h-1 w-1 rounded-full"
          style={{ background: colors.primary, boxShadow: `0 0 4px ${colors.primaryGlow}`, animation: "footer-pulse 2s ease-in-out infinite" }}
        />
        <span className="text-[8px] uppercase tracking-[0.28em]" style={{ color: colors.primaryTextDim }}>
          SYS:LOCMEMORY
        </span>
        <span className="text-[8px]" style={{ color: "rgba(0,196,188,0.15)" }}>·</span>
        <span className="text-[8px] uppercase tracking-[0.22em]" style={{ color: colors.primaryTextDim }}>
          BUILD 2025.04
        </span>
      </div>

      {/* Center — creators credit */}
      <NavLink
        to="/about"
        className="flex items-center gap-2 group"
        style={{ textDecoration: "none" }}
      >
        <span className="text-[9px] tracking-[0.15em]" style={{ color: colors.primaryTextDim }}>crafted by</span>
        <span
          className="text-[9px] uppercase tracking-[0.18em] font-semibold transition-all duration-300 group-hover:brightness-150"
          style={{ color: colors.primaryText }}
        >
          Almouthana Taha Khalfallah
        </span>
        <span className="text-[9px] font-bold" style={{ color: colors.primaryTextDim }}>&amp;</span>
        <span
          className="text-[9px] uppercase tracking-[0.18em] font-semibold transition-all duration-300 group-hover:brightness-150"
          style={{ color: "rgba(179,136,255,0.75)" }}
        >
          Eya Dhrif
        </span>
      </NavLink>

      {/* Right — live clock */}
      <div className="flex items-center gap-1.5">
        <span
          className="text-[8px] tabular-nums tracking-[0.15em]"
          style={{ color: colors.primaryTextDim, fontVariantNumeric: "tabular-nums" }}
        >
          {hh}:{mm}:{ss}
        </span>
        <div className="h-1 w-1 rounded-full" style={{ background: colors.primaryBorder }} />
      </div>

      <style>{`
        @keyframes footer-pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }
        .group:hover span { color: rgba(0,196,188,0.75) !important; }
      `}</style>
    </footer>
  )
}

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { colors } = useTheme()

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Mobile menu button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="fixed top-3 left-3 z-50 lg:hidden p-2 rounded-md"
        style={{ background: "rgba(0,5,16,0.9)", border: `1px solid ${colors.primaryBorder}` }}
      >
        {sidebarOpen ? (
          <X className="w-5 h-5" style={{ color: colors.primaryText }} />
        ) : (
          <Menu className="w-5 h-5" style={{ color: colors.primaryText }} />
        )}
      </button>

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 lg:hidden bg-black/50"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed lg:relative z-40 h-full transition-transform duration-300 ease-in-out
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
      `}>
        <Sidebar onNavigate={() => setSidebarOpen(false)} />
      </div>

      {/* Main content + footer */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
        <FooterBar />
      </div>
    </div>
  )
}
