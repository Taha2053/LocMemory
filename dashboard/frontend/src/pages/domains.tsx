import { useEffect, useState, useMemo } from "react"
import { api, type Domain } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { useNavigate } from "react-router-dom"
import { useTheme } from "@/lib/theme"

const DOMAIN_COLORS: Record<string, string> = {
  health:      "#00ff88",
  programming: "#00e5ff",
  work:        "#ff8c26",
  learning:    "#aaff00",
  personal:    "#c8a8ff",
  finance:     "#ffd700",
  engineering: "#ff6b6b",
  social:      "#38bdf8",
}

const FALLBACK_COLORS = [
  "#00ff88", "#00e5ff", "#ff8c26", "#aaff00",
  "#c8a8ff", "#ffd700", "#ff6b6b", "#38bdf8",
]

function domainColor(name: string, idx: number) {
  return DOMAIN_COLORS[name] ?? FALLBACK_COLORS[idx % FALLBACK_COLORS.length]
}

// ─── Stat Card ───────────────────────────────────────────────────────
function StatCard({ label, value, sub, color = "#00ff88" }: {
  label: string; value: string | number; sub?: string; color?: string
}) {
  return (
    <div className="px-4 py-3 rounded-sm"
      style={{ background: "rgba(0,5,16,0.7)", border: `1px solid ${color}20` }}>
      <div className="text-[8px] uppercase tracking-[0.25em] text-neutral-600 mb-1">{label}</div>
      <div className="text-[20px] font-bold tabular-nums font-mono"
        style={{ color, textShadow: `0 0 10px ${color}50` }}>
        {value}
      </div>
      {sub && <div className="text-[9px] text-neutral-600 mt-0.5 truncate">{sub}</div>}
    </div>
  )
}

// ─── Domain Row (left list) ──────────────────────────────────────────
function DomainRow({ domain, idx, total, isActive, onClick }: {
  domain: Domain; idx: number; total: number; isActive: boolean; onClick: () => void
}) {
  const color  = domainColor(domain.name, idx)
  const pct    = total > 0 ? (domain.total / total) * 100 : 0

  return (
    <button
      onClick={onClick}
      className="w-full text-left px-3 py-2.5 rounded-sm transition-all duration-150 group"
      style={{
        background: isActive ? `${color}0e` : "transparent",
        border: `1px solid ${isActive ? `${color}35` : "rgba(255,255,255,0.04)"}`,
        boxShadow: isActive ? `0 0 16px ${color}12` : "none",
      }}
    >
      <div className="flex items-center gap-2.5 mb-1.5">
        {/* Color dot */}
        <span className="w-2 h-2 rounded-full shrink-0 transition-all"
          style={{
            background: color,
            boxShadow: isActive ? `0 0 8px ${color}` : `0 0 3px ${color}80`,
          }} />

        {/* Name */}
        <span
          className="flex-1 text-[11px] uppercase tracking-wider font-medium capitalize transition-colors"
          style={{ color: isActive ? color : "#a3a3a3" }}
        >
          {domain.name}
        </span>

        {/* Count */}
        <span className="text-[10px] font-mono tabular-nums shrink-0"
          style={{ color: isActive ? color : "#525252" }}>
          {domain.total}
        </span>

        {/* Percent */}
        <span className="text-[9px] text-neutral-700 w-8 text-right shrink-0 tabular-nums">
          {pct.toFixed(0)}%
        </span>
      </div>

      {/* Bar */}
      <div className="h-[2px] w-full rounded-full overflow-hidden ml-4"
        style={{ background: "rgba(255,255,255,0.05)" }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: color,
            boxShadow: isActive ? `0 0 4px ${color}` : "none",
          }}
        />
      </div>
    </button>
  )
}

// ─── Domain Detail (right panel) ─────────────────────────────────────
function DomainDetail({ domain, idx, total, onBrowse }: {
  domain: Domain; idx: number; total: number; onBrowse: (name: string) => void
}) {
  const color       = domainColor(domain.name, idx)
  const pct         = total > 0 ? (domain.total / total) * 100 : 0
  const activeSubs  = domain.subdomains.filter((s) => s.count > 0).sort((a, b) => b.count - a.count)
  const maxSubCount = activeSubs.length > 0 ? activeSubs[0].count : 1

  return (
    <div className="h-full flex flex-col">
      {/* Domain header */}
      <div className="px-6 py-5" style={{ borderBottom: `1px solid ${color}20` }}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="text-[8px] uppercase tracking-[0.3em] text-neutral-600 mb-1">Selected domain</div>
            <h2
              className="text-[22px] font-bold uppercase tracking-wider capitalize mb-1"
              style={{ color, textShadow: `0 0 20px ${color}40` }}
            >
              {domain.name}
            </h2>
            <div className="flex items-center gap-3 text-[10px] text-neutral-500">
              <span><span className="text-neutral-300 font-mono">{domain.total}</span> memories</span>
              <span className="text-neutral-700">·</span>
              <span><span className="text-neutral-300 font-mono">{pct.toFixed(1)}%</span> of total</span>
              <span className="text-neutral-700">·</span>
              <span><span className="text-neutral-300 font-mono">{activeSubs.length}</span> subdomains</span>
            </div>
          </div>

          {/* Big percentage ring-ish indicator */}
          <div className="shrink-0 text-right">
            <div className="text-[32px] font-bold tabular-nums font-mono leading-none"
              style={{ color, textShadow: `0 0 16px ${color}60` }}>
              {pct.toFixed(0)}
              <span className="text-[16px] text-neutral-600">%</span>
            </div>
            <div className="text-[8px] uppercase tracking-widest text-neutral-700 mt-0.5">share</div>
          </div>
        </div>

        {/* Full-width progress bar */}
        <div className="mt-4 h-1.5 w-full rounded-full overflow-hidden"
          style={{ background: "rgba(255,255,255,0.06)" }}>
          <div
            className="h-full rounded-full transition-all duration-1000"
            style={{ width: `${pct}%`, background: color, boxShadow: `0 0 6px ${color}` }}
          />
        </div>
      </div>

      {/* Subdomains */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {activeSubs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-center">
            <div className="text-[10px] text-neutral-700 uppercase tracking-widest">No subdomains classified yet</div>
          </div>
        ) : (
          <div>
            <div className="text-[8px] uppercase tracking-[0.3em] text-neutral-600 mb-3">Subdomain breakdown</div>
            <div className="space-y-2.5">
              {activeSubs.map((sub, i) => {
                const subPct   = (sub.count / (domain.total || 1)) * 100
                const barWidth = (sub.count / maxSubCount) * 100
                const opacity  = 0.9 - i * 0.07
                return (
                  <div key={sub.name}>
                    <div className="flex items-center justify-between text-[10px] mb-1">
                      <div className="flex items-center gap-2">
                        <div className="h-px w-3 shrink-0" style={{ background: `${color}50` }} />
                        <span className="text-neutral-300 capitalize">{sub.name}</span>
                      </div>
                      <div className="flex items-center gap-2 text-neutral-600">
                        <span className="font-mono tabular-nums text-neutral-400">{sub.count}</span>
                        <span className="text-[9px] w-8 text-right">{subPct.toFixed(0)}%</span>
                      </div>
                    </div>
                    <div className="h-[3px] w-full rounded-full overflow-hidden"
                      style={{ background: "rgba(255,255,255,0.05)" }}>
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${barWidth}%`,
                          background: color,
                          opacity,
                          boxShadow: `0 0 4px ${color}60`,
                          transitionDelay: `${i * 40}ms`,
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Browse button */}
      <div className="px-6 py-4" style={{ borderTop: `1px solid rgba(255,255,255,0.05)` }}>
        <button
          onClick={() => onBrowse(domain.name)}
          className="w-full py-2.5 text-[10px] uppercase tracking-widest transition-all hover:opacity-90"
          style={{
            color,
            border: `1px solid ${color}35`,
            background: `${color}08`,
          }}
        >
          browse {domain.name} memories →
        </button>
      </div>
    </div>
  )
}

// ─── Treemap-style overview ──────────────────────────────────────────
function DomainBubbles({ domains, total, selected, onSelect }: {
  domains: Domain[]; total: number; selected: string | null; onSelect: (name: string) => void
}) {
  const sorted = [...domains].sort((a, b) => b.total - a.total)

  return (
    <div className="flex flex-wrap gap-2">
      {sorted.map((d, idx) => {
        const color   = domainColor(d.name, idx)
        const pct     = total > 0 ? (d.total / total) * 100 : 0
        const isActive = selected === d.name
        // Scale size: min 60px, max 130px
        const size    = Math.max(60, Math.min(130, 60 + (pct / 100) * 500))

        return (
          <button
            key={d.name}
            onClick={() => onSelect(d.name)}
            className="flex flex-col items-center justify-center rounded-sm transition-all duration-200 shrink-0"
            style={{
              width: size,
              height: size * 0.75,
              background: isActive ? `${color}15` : "rgba(0,5,16,0.7)",
              border: `1px solid ${isActive ? color : `${color}25`}`,
              boxShadow: isActive ? `0 0 20px ${color}25` : "none",
            }}
          >
            <span className="text-[10px] uppercase tracking-wider capitalize font-medium mb-1"
              style={{ color: isActive ? color : `${color}90` }}>
              {d.name}
            </span>
            <span className="text-[11px] font-bold font-mono tabular-nums"
              style={{ color: isActive ? color : "#525252" }}>
              {d.total}
            </span>
            <span className="text-[8px] text-neutral-700 tabular-nums">{pct.toFixed(0)}%</span>
          </button>
        )
      })}
    </div>
  )
}

// ─── Main Page ───────────────────────────────────────────────────────
export function DomainsPage() {
  const [domains, setDomains]   = useState<Domain[]>([])
  const [loading, setLoading]   = useState(true)
  const [selected, setSelected] = useState<string | null>(null)
  const [search, setSearch]     = useState("")
  const navigate = useNavigate()

  useEffect(() => {
    api.domains()
      .then((data) => {
        const active = data.filter((d) => d.total > 0).sort((a, b) => b.total - a.total)
        setDomains(active)
        if (active.length > 0) setSelected(active[0].name)
      })
      .finally(() => setLoading(false))
  }, [])

  const total = useMemo(() => domains.reduce((a, d) => a + d.total, 0), [domains])

  const topDomain   = domains[0]
  const classified  = domains.filter((d) => d.name !== "(none)").reduce((a, d) => a + d.total, 0)
  const classifiedPct = total > 0 ? ((classified / total) * 100).toFixed(0) : "0"

  const filtered = useMemo(() =>
    search
      ? domains.filter((d) => d.name.toLowerCase().includes(search.toLowerCase()))
      : domains,
    [domains, search]
  )

  const selectedDomain = domains.find((d) => d.name === selected)
  const selectedIdx    = domains.findIndex((d) => d.name === selected)

  const handleBrowse = (domainName: string) => {
    navigate(`/memories?domain=${domainName}`)
  }

  const { theme } = useTheme()

  return (
    <div className={`relative flex flex-col h-full min-h-0 font-mono overflow-hidden ${theme === "dark" ? "bg-[#020d0d]" : "bg-slate-50"}`}>
      {theme === "dark" && <ScanlineOverlay />}

      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 20% 0%, rgba(0,255,136,0.05), transparent 50%), radial-gradient(ellipse at 80% 100%, rgba(0,229,255,0.04), transparent 50%)" }} />

      <div className="relative z-10 flex flex-col h-full min-h-0">

        {/* ── Top header + stats ── */}
        <div className="px-6 py-5 shrink-0" style={{ borderBottom: "1px solid rgba(0,255,136,0.1)" }}>
          <div className="flex items-start justify-between gap-6 mb-5">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <div className="h-px w-6 bg-emerald-400/40" />
                <span className="text-[8px] uppercase tracking-[0.3em] text-emerald-600/60">Knowledge Domains</span>
              </div>
              <h1 className="text-[20px] font-bold tracking-wide text-emerald-300"
                style={{ textShadow: "0 0 16px rgba(0,255,136,0.4)" }}>
                Knowledge Domains
              </h1>
              <p className="text-[10px] text-neutral-600 mt-0.5 uppercase tracking-wider">
                Auto-classified by semantic embedding model
              </p>
            </div>

            {/* Stats row */}
            {!loading && (
              <div className="flex gap-3 shrink-0">
                <StatCard label="Domains" value={domains.length} color="#00ff88" />
                <StatCard label="Memories" value={total.toLocaleString()} color="#00e5ff" />
                <StatCard label="Classified" value={`${classifiedPct}%`} color="#aaff00" sub="of all memories" />
                <StatCard
                  label="Top domain"
                  value={topDomain?.name ?? "—"}
                  sub={`${topDomain?.total ?? 0} memories`}
                  color="#ff8c26"
                />
              </div>
            )}
          </div>

          {/* Bubble overview */}
          {!loading && domains.length > 0 && (
            <DomainBubbles
              domains={domains}
              total={total}
              selected={selected}
              onSelect={setSelected}
            />
          )}
        </div>

        {/* ── Body: list + detail ── */}
        <div className="flex flex-1 min-h-0">

          {/* Left: domain list */}
          <div
            className="w-64 shrink-0 flex flex-col overflow-hidden"
            style={{ borderRight: "1px solid rgba(0,255,136,0.1)" }}
          >
            {/* Search */}
            <div className="px-3 py-2.5" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="filter domains..."
                className="w-full bg-transparent text-[11px] text-neutral-300 placeholder:text-neutral-700 outline-none"
              />
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
              {loading ? (
                <div className="text-[10px] text-emerald-500/40 animate-pulse uppercase tracking-widest py-6 text-center">
                  loading...
                </div>
              ) : filtered.length === 0 ? (
                <div className="text-[10px] text-neutral-700 uppercase tracking-widest py-6 text-center">
                  no domains
                </div>
              ) : (
                filtered.map((d, idx) => (
                  <DomainRow
                    key={d.name}
                    domain={d}
                    idx={domains.indexOf(d)}
                    total={total}
                    isActive={selected === d.name}
                    onClick={() => setSelected(d.name)}
                  />
                ))
              )}
            </div>

            {/* Footer */}
            <div className="px-3 py-2.5 flex items-center justify-between"
              style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
              <span className="text-[8px] uppercase tracking-widest text-neutral-700">
                {filtered.length} domains
              </span>
              <span className="text-[8px] font-mono text-neutral-700 tabular-nums">
                {total} total
              </span>
            </div>
          </div>

          {/* Right: domain detail */}
          <div className="flex-1 min-w-0 overflow-hidden">
            {selectedDomain ? (
              <DomainDetail
                domain={selectedDomain}
                idx={selectedIdx}
                total={total}
                onBrowse={handleBrowse}
              />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-[10px] uppercase tracking-widest text-neutral-700">
                  select a domain
                </div>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}
