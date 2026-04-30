import { useEffect, useState } from "react"
import { api, type Domain } from "@/lib/api"
import { ChevronRight, ChevronDown } from "lucide-react"
import { ScanlineOverlay } from "@/components/hud"

const TIER_COLORS = ["#3b82f6", "#06b6d4", "#9ec5e8", "#a855f7"] as const

export function DomainsPage() {
  const [domains, setDomains] = useState<Domain[]>([])
  const [open, setOpen] = useState<Record<string, boolean>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.domains()
      .then(setDomains)
      .finally(() => setLoading(false))
  }, [])

  const toggle = (name: string) =>
    setOpen((s) => ({ ...s, [name]: !s[name] }))

  const totalMemories = domains.reduce((a, d) => a + d.total, 0)

  return (
    <div className="relative h-full min-h-0 bg-[#000510] font-mono overflow-y-auto">
      <ScanlineOverlay />

      {/* Ambient corner glow */}
      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 0% 0%, rgba(59,130,246,0.08), transparent 40%), radial-gradient(ellipse at 100% 100%, rgba(6,182,212,0.06), transparent 40%)" }} />

      {/* Corner brackets */}
      <div className="pointer-events-none absolute top-3 left-3 h-5 w-5 border-t-2 border-l-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />
      <div className="pointer-events-none absolute top-3 right-3 h-5 w-5 border-t-2 border-r-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-cyan-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(34,211,238,0.5))" }} />

      <div className="relative z-10 max-w-3xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px w-8 bg-cyan-400/40" style={{ boxShadow: "0 0 4px rgba(34,211,238,0.4)" }} />
            <span className="text-[9px] uppercase tracking-[0.3em] text-cyan-600/60">// SYS.DOMAINS</span>
          </div>
          <h1 className="text-2xl font-bold tracking-wide text-cyan-300"
            style={{ textShadow: "0 0 20px rgba(34,211,238,0.4)" }}>
            Knowledge Domains
          </h1>
          <p className="mt-1 text-[11px] text-neutral-500 uppercase tracking-wider">
            LLM-classified categories — {totalMemories.toLocaleString()} memories indexed
          </p>
        </div>

        {/* Stats row */}
        <div className="mb-6 grid grid-cols-3 gap-3">
          {[
            { label: "DOMAINS",   value: domains.length },
            { label: "MEMORIES",  value: totalMemories },
            { label: "AVG / DOM", value: domains.length ? Math.round(totalMemories / domains.length) : 0 },
          ].map(({ label, value }) => (
            <div key={label}
              className="px-4 py-3 border border-cyan-400/15 rounded-sm"
              style={{ background: "rgba(6,182,212,0.04)", boxShadow: "0 0 20px rgba(6,182,212,0.06)" }}>
              <div className="text-[8px] uppercase tracking-[0.25em] text-neutral-600 mb-1">{label}</div>
              <div className="text-xl font-bold tabular-nums text-cyan-300"
                style={{ textShadow: "0 0 8px rgba(34,211,238,0.5)" }}>
                {value.toLocaleString()}
              </div>
            </div>
          ))}
        </div>

        {/* Domain list */}
        {loading ? (
          <div className="text-[10px] uppercase tracking-widest text-cyan-500/50 animate-pulse py-8 text-center">
            LOADING DOMAINS...
          </div>
        ) : (
          <div className="space-y-2">
            {domains.map((d, idx) => {
              const color = TIER_COLORS[idx % 4]
              const isOpen = open[d.name] ?? false
              const pct = totalMemories > 0 ? (d.total / totalMemories) * 100 : 0

              return (
                <div key={d.name}
                  className="border border-white/5 rounded-sm overflow-hidden transition-all duration-200"
                  style={{
                    background: isOpen ? "rgba(6,182,212,0.04)" : "rgba(0,5,16,0.6)",
                    borderColor: isOpen ? `${color}30` : "rgba(255,255,255,0.05)",
                    boxShadow: isOpen ? `0 0 20px ${color}15` : "none",
                  }}>

                  <button
                    className="w-full flex items-center gap-3 px-4 py-3 text-left transition-all duration-150 group"
                    onClick={() => toggle(d.name)}
                  >
                    {/* Color accent bar */}
                    <div className="shrink-0 h-4 w-0.5 rounded-full"
                      style={{ background: color, boxShadow: `0 0 6px ${color}` }} />

                    {/* Chevron */}
                    <div className="shrink-0 transition-transform duration-200"
                      style={{ color, transform: isOpen ? "rotate(0deg)" : "rotate(-90deg)" }}>
                      <ChevronDown className="w-3.5 h-3.5" />
                    </div>

                    {/* Name */}
                    <span className="flex-1 text-[12px] uppercase tracking-wider font-medium transition-colors"
                      style={{ color: isOpen ? color : "#a3a3a3", textShadow: isOpen ? `0 0 8px ${color}50` : "none" }}>
                      {d.name}
                    </span>

                    {/* Progress bar */}
                    <div className="w-24 h-1 bg-white/5 rounded-full overflow-hidden hidden sm:block">
                      <div className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, background: color, boxShadow: `0 0 4px ${color}` }} />
                    </div>

                    {/* Count badge */}
                    <span className="text-[10px] font-mono tabular-nums px-2 py-0.5 rounded-sm"
                      style={{ color, background: `${color}18`, border: `1px solid ${color}30` }}>
                      {d.total.toLocaleString()}
                    </span>
                  </button>

                  {/* Subdomains */}
                  <div className={`overflow-hidden transition-all duration-300 ${isOpen ? "max-h-96" : "max-h-0"}`}>
                    <div className="border-t px-4 py-2 space-y-0.5"
                      style={{ borderColor: `${color}20` }}>
                      {d.subdomains.length === 0 ? (
                        <div className="py-2 text-[10px] text-neutral-700 uppercase tracking-widest">
                          NO SUBDOMAINS
                        </div>
                      ) : (
                        d.subdomains.map((s) => (
                          <div key={s.name}
                            className="flex items-center justify-between px-2 py-1.5 rounded-sm hover:bg-white/5 transition-colors group/sub">
                            <div className="flex items-center gap-2">
                              <div className="h-px w-4" style={{ background: `${color}40` }} />
                              <span className="text-[10px] text-neutral-500 group-hover/sub:text-neutral-300 uppercase tracking-wider transition-colors">
                                {s.name}
                              </span>
                            </div>
                            <span className="text-[9px] font-mono tabular-nums text-neutral-600">{s.count}</span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
