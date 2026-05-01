import { useEffect, useState, useCallback } from "react"
import { api, type Memory, type MemoryDetail, type Domain } from "@/lib/api"
import { ChevronDown, ChevronRight, X } from "lucide-react"
import { ScanlineOverlay } from "@/components/hud"

const TIER_COLORS = ["#00c4bc", "#ff8c26", "#ffd700", "#ff4d6d"] as const

const TYPEWRITER_TEXT = "SEARCH MEMORIES..."
let typewriterInterval: ReturnType<typeof setInterval> | null = null

export function MemoriesPage() {
  const [memories, setMemories] = useState<Memory[]>([])
  const [domains, setDomains] = useState<Domain[]>([])
  const [selectedMemory, setSelectedMemory] = useState<MemoryDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [search, setSearch] = useState("")
  const [activeDomain, setActiveDomain] = useState<string | null>(null)
  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set())
  const [page, setPage] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const [typewriterText, setTypewriterText] = useState("")

  useEffect(() => {
    api.domains().then(setDomains).catch(() => {})
  }, [])

  const loadMemories = useCallback(
    (reset = false) => {
      const pageNum = reset ? 0 : page
      const params: { domain?: string; q?: string } = {}
      if (activeDomain) params.domain = activeDomain
      if (search) params.q = search
      const loader = reset ? setLoading : setLoadingMore
      loader(true)
      api.memories({ ...params, limit: 20, offset: pageNum * 20 } as any)
        .then((data) => {
          reset ? setMemories(data) : setMemories((prev) => [...prev, ...data])
          setHasMore(data.length === 20)
        })
        .finally(() => loader(false))
    },
    [page, activeDomain, search]
  )

  useEffect(() => {
    setPage(0)
    loadMemories(true)
  }, [activeDomain, search])

  useEffect(() => {
    if (!search && !activeDomain) {
      let idx = 0
      typewriterInterval = setInterval(() => {
        setTypewriterText(TYPEWRITER_TEXT.slice(0, idx + 1))
        if (++idx >= TYPEWRITER_TEXT.length) idx = 0
      }, 100)
      return () => { if (typewriterInterval) clearInterval(typewriterInterval) }
    } else {
      setTypewriterText("")
    }
  }, [search, activeDomain])

  const loadMore = () => {
    if (!loadingMore && hasMore) {
      setPage((p) => p + 1)
      loadMemories(false)
    }
  }

  const selectMemory = async (id: string) => {
    try {
      const detail = await api.memory(id)
      setSelectedMemory(detail)
    } catch (e) {
      console.error("Failed to load memory detail", e)
    }
  }

  const toggleDomain = (name: string) => {
    setExpandedDomains((prev) => {
      const next = new Set(prev)
      next.has(name) ? next.delete(name) : next.add(name)
      return next
    })
  }

  const getTierColor = (tier: number) => TIER_COLORS[tier] || TIER_COLORS[0]

  return (
    <div className="relative flex h-full min-h-0 bg-[#020d0d] font-mono">
      <ScanlineOverlay />

      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-0 z-0"
        style={{ background: "radial-gradient(ellipse at 0% 50%, rgba(59, 200, 215,0.06), transparent 40%)" }} />

      {/* ── Domains sidebar ── */}
      <div
        className="relative z-10 w-52 shrink-0 flex flex-col overflow-y-auto"
        style={{ borderRight: "1px solid rgba(0, 196, 188,0.12)", background: "rgba(0,5,16,0.8)" }}
      >
        <div className="px-4 py-4" style={{ borderBottom: "1px solid rgba(0, 196, 188,0.1)" }}>
          <div className="flex items-center gap-2 mb-0.5">
            <div className="h-px w-4" style={{ background: "rgba(0, 196, 188,0.4)" }} />
            <span className="text-[8px] uppercase tracking-[0.25em] text-emerald-600/60">// FILTER</span>
          </div>
          <div className="text-[11px] uppercase tracking-wider text-neutral-400">Domains</div>
        </div>

        <div className="flex-1 px-2 py-3 space-y-0.5">
          {domains.map((domain, idx) => {
            const color = TIER_COLORS[idx % 4]
            const isActive = activeDomain === domain.name
            const isExpanded = expandedDomains.has(domain.name)
            return (
              <div key={domain.name}>
                <button
                  onClick={() => {
                    setActiveDomain(isActive ? null : domain.name)
                    toggleDomain(domain.name)
                  }}
                  className="w-full flex items-center gap-2 px-2 py-1.5 text-left text-[10px] uppercase tracking-wider transition-all rounded-sm"
                  style={{
                    color: isActive ? color : "#737373",
                    background: isActive ? `${color}12` : "transparent",
                    borderLeft: isActive ? `2px solid ${color}` : "2px solid transparent",
                  }}
                >
                  {isExpanded || isActive
                    ? <ChevronDown className="w-3 h-3 shrink-0" />
                    : <ChevronRight className="w-3 h-3 shrink-0" />}
                  <span className="flex-1 truncate" style={{ textShadow: isActive ? `0 0 6px ${color}60` : "none" }}>
                    {domain.name}
                  </span>
                  <span className="text-[9px] font-mono px-1 py-0.5 rounded-sm"
                    style={{ color, background: `${color}18` }}>
                    {domain.total}
                  </span>
                </button>

                <div className={`ml-4 overflow-hidden transition-all ${isExpanded || isActive ? "max-h-40" : "max-h-0"}`}>
                  {domain.subdomains?.map((sub) => (
                    <button
                      key={sub.name}
                      onClick={() => setActiveDomain(sub.name)}
                      className="w-full text-left px-2 py-1 text-[9px] text-neutral-600 hover:text-emerald-300 transition-colors flex items-center gap-1.5"
                    >
                      <div className="h-px w-3" style={{ background: `${color}30` }} />
                      <span className="truncate">{sub.name}</span>
                      <span className="ml-auto text-neutral-700">{sub.count}</span>
                    </button>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        {activeDomain && (
          <div className="px-3 py-3" style={{ borderTop: "1px solid rgba(0, 196, 188,0.08)" }}>
            <button
              onClick={() => setActiveDomain(null)}
              className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest text-neutral-600 hover:text-emerald-400 transition-colors"
            >
              <X className="w-3 h-3" /> CLEAR FILTER
            </button>
          </div>
        )}
      </div>

      {/* ── Main content ── */}
      <div className="relative z-10 flex-1 flex flex-col min-w-0 px-5 py-5">

        {/* Search bar */}
        <div className="relative mb-5">
          <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-emerald-400/40"
            style={{ boxShadow: "0 0 6px rgba(0, 196, 188,0.4)" }} />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={typewriterText || TYPEWRITER_TEXT}
            className="w-full bg-transparent py-2.5 pl-4 pr-24 text-[12px] text-neutral-100 placeholder:text-neutral-700 focus:outline-none transition-all"
            style={{ borderBottom: "1px solid rgba(0, 196, 188,0.2)" }}
            onFocus={(e) => (e.target.style.borderBottomColor = "rgba(0, 196, 188,0.5)")}
            onBlur={(e) => (e.target.style.borderBottomColor = "rgba(0, 196, 188,0.2)")}
          />
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
            {(activeDomain || search) && (
              <span className="text-[9px] text-emerald-400 bg-emerald-400/10 px-1.5 py-0.5 rounded-sm">
                {memories.length} results
              </span>
            )}
            <span className="text-[8px] text-neutral-700 border border-neutral-800 px-1.5 py-0.5">⌘K</span>
          </div>
        </div>

        {/* Memory grid */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="text-[10px] uppercase tracking-widest text-emerald-500/50 animate-pulse py-12 text-center">
              SYNCING MEMORY GRAPH...
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
                {memories.map((mem) => {
                  const color = getTierColor(mem.tier)
                  return (
                    <button
                      key={mem.id}
                      onClick={() => selectMemory(mem.id)}
                      className="group text-left rounded-sm overflow-hidden transition-all duration-150 hover:scale-[1.005]"
                      style={{
                        background: "rgba(0,5,16,0.7)",
                        border: "1px solid rgba(255,255,255,0.05)",
                        borderLeft: `2px solid ${color}`,
                      }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.boxShadow = `0 0 20px ${color}20`
                        ;(e.currentTarget as HTMLElement).style.borderColor = `${color}60`
                        ;(e.currentTarget as HTMLElement).style.borderLeftColor = color
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.boxShadow = "none"
                        ;(e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.05)"
                        ;(e.currentTarget as HTMLElement).style.borderLeftColor = color
                      }}
                    >
                      <div className="px-3 pt-3 pb-2">
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="w-1.5 h-1.5 rounded-full shrink-0"
                            style={{ background: color, boxShadow: `0 0 4px ${color}` }} />
                          <span className="text-[9px] uppercase tracking-wider text-neutral-500">{mem.tier_name}</span>
                          {mem.domain && (
                            <>
                              <span className="text-neutral-700 text-[9px]">›</span>
                              <span className="text-[9px] text-neutral-600 truncate">{mem.domain}</span>
                            </>
                          )}
                        </div>
                        <p className="text-[11px] text-neutral-300 line-clamp-2 leading-relaxed">{mem.text}</p>
                      </div>
                      <div className="px-3 py-1.5 flex items-center justify-between border-t"
                        style={{ borderColor: "rgba(255,255,255,0.04)" }}>
                        <span className="text-[8px] text-neutral-700 font-mono">{mem.created_at?.slice(0, 10) || "—"}</span>
                        <span className="text-[8px] uppercase tracking-wider text-emerald-500/0 group-hover:text-emerald-500/70 transition-colors">
                          VIEW →
                        </span>
                      </div>
                    </button>
                  )
                })}
              </div>

              {hasMore && (
                <button
                  onClick={loadMore}
                  disabled={loadingMore}
                  className="w-full mt-4 py-2.5 text-[10px] uppercase tracking-widest border transition-all"
                  style={{
                    borderColor: "rgba(0, 196, 188,0.2)",
                    color: loadingMore ? "rgba(0, 196, 188,0.3)" : "rgba(0, 196, 188,0.6)",
                  }}
                >
                  {loadingMore ? "LOADING..." : "LOAD MORE"}
                </button>
              )}

              {memories.length === 0 && (
                <div className="text-center text-[11px] text-neutral-700 uppercase tracking-widest py-16">
                  NO MATCHING MEMORIES
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* ── Detail panel ── */}
      <div
        className="hidden 2xl:flex relative z-10 w-64 shrink-0 flex-col overflow-y-auto"
        style={{ borderLeft: "1px solid rgba(0, 196, 188,0.12)", background: "rgba(0,5,16,0.8)" }}
      >
        <div className="px-4 py-4" style={{ borderBottom: "1px solid rgba(0, 196, 188,0.1)" }}>
          <div className="flex items-center gap-2 mb-0.5">
            <div className="h-px w-4" style={{ background: "rgba(0, 196, 188,0.4)" }} />
            <span className="text-[8px] uppercase tracking-[0.25em] text-emerald-600/60">// DETAIL</span>
          </div>
          <div className="text-[11px] uppercase tracking-wider text-neutral-400">Memory Inspector</div>
        </div>

        <div className="flex-1 px-4 py-4">
          {selectedMemory ? (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <span className="w-2 h-2 rounded-full"
                  style={{ background: getTierColor(selectedMemory.tier), boxShadow: `0 0 6px ${getTierColor(selectedMemory.tier)}` }} />
                <span className="text-[10px] uppercase tracking-wider text-neutral-400">{selectedMemory.tier_name}</span>
              </div>
              <p className="text-[11px] text-neutral-200 leading-relaxed mb-4 whitespace-pre-wrap">{selectedMemory.text}</p>

              <div className="space-y-2 text-[9px] mb-4">
                {[
                  { label: "DOMAIN",    value: selectedMemory.domain },
                  { label: "SUBDOMAIN", value: selectedMemory.subdomain },
                  { label: "CREATED",   value: selectedMemory.created_at?.slice(0, 10) },
                  { label: "NEIGHBORS", value: String(selectedMemory.neighbors?.length || 0) },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between py-1"
                    style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <span className="text-neutral-600 uppercase tracking-wider">{label}</span>
                    <span className="text-neutral-400 font-mono">{value || "—"}</span>
                  </div>
                ))}
              </div>

              {selectedMemory.neighbors && selectedMemory.neighbors.length > 0 && (
                <div className="pt-3" style={{ borderTop: "1px solid rgba(0, 196, 188,0.1)" }}>
                  <div className="text-[8px] uppercase tracking-[0.25em] text-emerald-600/60 mb-2">// RELATED</div>
                  <div className="space-y-1.5">
                    {selectedMemory.neighbors.slice(0, 5).map((n) => (
                      <button
                        key={n.id}
                        className="w-full text-left text-[9px] text-neutral-500 truncate hover:text-emerald-300 transition-colors py-0.5"
                        onClick={() => selectMemory(n.id)}
                      >
                        {n.text}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <button
                onClick={() => setSelectedMemory(null)}
                className="mt-5 flex items-center gap-1 text-[9px] uppercase tracking-widest text-neutral-600 hover:text-emerald-400 transition-colors"
              >
                <X className="w-3 h-3" /> CLOSE
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center py-12">
              <div className="text-[8px] uppercase tracking-[0.25em] text-neutral-700 mb-2">// AWAITING</div>
              <div className="text-[10px] text-neutral-700 uppercase tracking-widest">Select a memory</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
