import { useEffect, useState, useCallback, useRef } from "react"
import { useSearchParams } from "react-router-dom"
import { api, type Memory, type MemoryDetail, type Domain } from "@/lib/api"
import { ChevronDown, ChevronRight, X, Plus, Pencil, Trash2, Copy, Check, ArrowUpDown } from "lucide-react"
import { ScanlineOverlay } from "@/components/hud"
import { useTheme } from "@/lib/theme"

const TIER_COLORS = ["#00ff88", "#00e5ff", "#aaff00", "#00ff66"] as const
const TIER_LABELS = [
  { tier: 1, label: "Context" },
  { tier: 2, label: "Anchor" },
  { tier: 3, label: "Leaf" },
  { tier: 4, label: "Procedural" },
]
const DOMAINS_LIST = ["health", "programming", "work", "learning", "personal", "finance", "engineering", "social"]
const SORT_OPTIONS = [
  { value: "newest", label: "Newest first" },
  { value: "oldest", label: "Oldest first" },
  { value: "tier-asc", label: "Tier ↑" },
  { value: "tier-desc", label: "Tier ↓" },
]

const TYPEWRITER_TEXT = "SEARCH MEMORIES..."

function getTierColor(tier: number) {
  return TIER_COLORS[Math.max(0, tier - 1)] ?? TIER_COLORS[0]
}

// ─── Add Memory Modal ───────────────────────────────────────────────
function AddMemoryModal({
  onClose,
  onCreated,
}: {
  onClose: () => void
  onCreated: (mem: Memory) => void
}) {
  const [text, setText] = useState("")
  const [tier, setTier] = useState(3)
  const [domain, setDomain] = useState("")
  const [subdomain, setSubdomain] = useState("")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const textRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => { textRef.current?.focus() }, [])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim()) { setError("Text is required"); return }
    setSaving(true)
    setError(null)
    try {
      const mem = await api.createMemory({
        text: text.trim(),
        tier,
        domain: domain || undefined,
        subdomain: subdomain || undefined,
      })
      onCreated(mem)
    } catch {
      setError("Failed to create memory — is the backend running?")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.7)", backdropFilter: "blur(4px)" }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="w-[520px] max-w-[95vw] font-mono"
        style={{
          background: "rgba(2,13,13,0.98)",
          border: "1px solid rgba(0,255,136,0.25)",
          borderRadius: "8px",
          boxShadow: "0 0 60px rgba(0,255,136,0.08)",
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4"
          style={{ borderBottom: "1px solid rgba(0,255,136,0.1)" }}>
          <div>
            <div className="text-[8px] uppercase tracking-[0.3em] text-emerald-600/60 mb-0.5">// NEW</div>
            <div className="text-[13px] uppercase tracking-widest text-emerald-300">Add Memory</div>
          </div>
          <button onClick={onClose} className="text-neutral-600 hover:text-neutral-300 transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        <form onSubmit={submit} className="px-5 py-5 space-y-4">
          {/* Text */}
          <div>
            <label className="block text-[9px] uppercase tracking-widest text-neutral-600 mb-1.5">
              Memory text *
            </label>
            <textarea
              ref={textRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              placeholder="Enter the fact or memory to store..."
              className="w-full bg-black/40 text-[12px] text-neutral-100 placeholder:text-neutral-700 resize-none outline-none px-3 py-2.5 transition-colors"
              style={{
                border: "1px solid rgba(0,255,136,0.15)",
                borderRadius: "4px",
              }}
              onFocus={(e) => (e.target.style.borderColor = "rgba(0,255,136,0.4)")}
              onBlur={(e) => (e.target.style.borderColor = "rgba(0,255,136,0.15)")}
            />
          </div>

          {/* Tier */}
          <div>
            <label className="block text-[9px] uppercase tracking-widest text-neutral-600 mb-1.5">Tier</label>
            <div className="flex gap-2">
              {TIER_LABELS.map(({ tier: t, label }) => {
                const color = getTierColor(t)
                const isActive = tier === t
                return (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setTier(t)}
                    className="flex-1 py-1.5 text-[9px] uppercase tracking-wider transition-all rounded-sm"
                    style={{
                      color: isActive ? color : "#525252",
                      background: isActive ? `${color}15` : "transparent",
                      border: `1px solid ${isActive ? color : "rgba(255,255,255,0.08)"}`,
                    }}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Domain + Subdomain */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[9px] uppercase tracking-widest text-neutral-600 mb-1.5">
                Domain <span className="text-neutral-700">(auto if empty)</span>
              </label>
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                className="w-full bg-black/40 text-[11px] text-neutral-300 outline-none px-3 py-2 transition-colors"
                style={{ border: "1px solid rgba(0,255,136,0.15)", borderRadius: "4px" }}
              >
                <option value="">auto-detect</option>
                {DOMAINS_LIST.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[9px] uppercase tracking-widest text-neutral-600 mb-1.5">Subdomain</label>
              <input
                type="text"
                value={subdomain}
                onChange={(e) => setSubdomain(e.target.value)}
                placeholder="optional"
                className="w-full bg-black/40 text-[11px] text-neutral-300 placeholder:text-neutral-700 outline-none px-3 py-2 transition-colors"
                style={{ border: "1px solid rgba(0,255,136,0.15)", borderRadius: "4px" }}
              />
            </div>
          </div>

          {error && (
            <div className="text-[10px] text-red-400/80">{error}</div>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-1">
            <button
              type="submit"
              disabled={saving || !text.trim()}
              className="flex-1 py-2 text-[10px] uppercase tracking-widest transition-all disabled:opacity-40"
              style={{
                background: "rgba(0,255,136,0.1)",
                border: "1px solid rgba(0,255,136,0.3)",
                color: "#00ff88",
              }}
            >
              {saving ? "saving..." : "[ save memory ]"}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-[10px] uppercase tracking-widest text-neutral-600 hover:text-neutral-300 transition-colors"
              style={{ border: "1px solid rgba(255,255,255,0.08)" }}
            >
              cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// ─── Detail Panel ───────────────────────────────────────────────────
function DetailPanel({
  memory,
  onClose,
  onUpdated,
  onDeleted,
}: {
  memory: MemoryDetail
  onClose: () => void
  onUpdated: (mem: Memory) => void
  onDeleted: (id: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState(memory.text)
  const [saving, setSaving] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [copied, setCopied] = useState(false)
  const color = getTierColor(memory.tier)

  const save = async () => {
    if (!editText.trim() || editText === memory.text) { setEditing(false); return }
    setSaving(true)
    try {
      const updated = await api.updateMemory(memory.id, editText.trim())
      onUpdated(updated)
      setEditing(false)
    } catch {
      // keep editing open on error
    } finally {
      setSaving(false)
    }
  }

  const del = async () => {
    setDeleting(true)
    try {
      await api.deleteMemory(memory.id)
      onDeleted(memory.id)
    } catch {
      setDeleting(false)
      setConfirmDelete(false)
    }
  }

  const copy = () => {
    navigator.clipboard.writeText(memory.text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div
      className="relative z-10 w-72 shrink-0 flex flex-col overflow-y-auto"
      style={{ borderLeft: "1px solid rgba(0,255,136,0.12)", background: "rgba(0,5,16,0.9)" }}
    >
      {/* Header */}
      <div className="px-4 py-3 flex items-center justify-between"
        style={{ borderBottom: "1px solid rgba(0,255,136,0.1)" }}>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full shrink-0"
            style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
          <span className="text-[10px] uppercase tracking-wider text-neutral-400">{memory.tier_name}</span>
        </div>
        <div className="flex items-center gap-1">
          {/* Copy */}
          <button
            onClick={copy}
            title="Copy text"
            className="p-1.5 text-neutral-600 hover:text-emerald-400 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
          {/* Edit toggle */}
          <button
            onClick={() => { setEditing(!editing); setEditText(memory.text); setConfirmDelete(false) }}
            title="Edit"
            className={`p-1.5 transition-colors ${editing ? "text-emerald-400" : "text-neutral-600 hover:text-emerald-400"}`}
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
          {/* Delete */}
          <button
            onClick={() => { setConfirmDelete(!confirmDelete); setEditing(false) }}
            title="Delete"
            className={`p-1.5 transition-colors ${confirmDelete ? "text-red-400" : "text-neutral-600 hover:text-red-400"}`}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
          {/* Close */}
          <button onClick={onClose} className="p-1.5 text-neutral-600 hover:text-neutral-300 transition-colors ml-1">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex-1 px-4 py-4 space-y-4 overflow-y-auto">

        {/* Delete confirmation */}
        {confirmDelete && (
          <div className="rounded-sm px-3 py-3 space-y-2.5"
            style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)" }}>
            <div className="text-[10px] text-red-400 uppercase tracking-wider">Delete this memory?</div>
            <p className="text-[9px] text-neutral-500">This cannot be undone.</p>
            <div className="flex gap-2">
              <button
                onClick={del}
                disabled={deleting}
                className="flex-1 py-1.5 text-[9px] uppercase tracking-wider text-red-400 transition-all disabled:opacity-40"
                style={{ border: "1px solid rgba(239,68,68,0.3)" }}
              >
                {deleting ? "deleting..." : "yes, delete"}
              </button>
              <button
                onClick={() => setConfirmDelete(false)}
                className="flex-1 py-1.5 text-[9px] uppercase tracking-wider text-neutral-500 hover:text-neutral-300 transition-colors"
                style={{ border: "1px solid rgba(255,255,255,0.08)" }}
              >
                cancel
              </button>
            </div>
          </div>
        )}

        {/* Memory text — editable or display */}
        {editing ? (
          <div className="space-y-2">
            <label className="text-[8px] uppercase tracking-widest text-neutral-600">Editing text</label>
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              rows={5}
              autoFocus
              className="w-full bg-black/40 text-[11px] text-neutral-100 resize-none outline-none px-3 py-2.5"
              style={{ border: "1px solid rgba(0,255,136,0.3)", borderRadius: "4px" }}
            />
            <div className="flex gap-2">
              <button
                onClick={save}
                disabled={saving}
                className="flex-1 py-1.5 text-[9px] uppercase tracking-wider transition-all disabled:opacity-40"
                style={{ color: "#00ff88", border: "1px solid rgba(0,255,136,0.3)" }}
              >
                {saving ? "saving..." : "save"}
              </button>
              <button
                onClick={() => setEditing(false)}
                className="flex-1 py-1.5 text-[9px] uppercase tracking-wider text-neutral-500 hover:text-neutral-300"
                style={{ border: "1px solid rgba(255,255,255,0.08)" }}
              >
                cancel
              </button>
            </div>
          </div>
        ) : (
          <p className="text-[11px] text-neutral-200 leading-relaxed whitespace-pre-wrap">{memory.text}</p>
        )}

        {/* Metadata */}
        <div className="space-y-0 text-[9px]" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
          {[
            { label: "Domain",    value: memory.domain },
            { label: "Subdomain", value: memory.subdomain },
            { label: "Created",   value: memory.created_at?.slice(0, 10) },
            { label: "ID",        value: memory.id.slice(0, 12) + "..." },
            { label: "Neighbors", value: String(memory.neighbors?.length || 0) + " nodes" },
          ].map(({ label, value }) => (
            <div key={label} className="flex justify-between py-1.5"
              style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
              <span className="text-neutral-600 uppercase tracking-wider">{label}</span>
              <span className="text-neutral-400 font-mono truncate max-w-[120px] text-right">{value || "—"}</span>
            </div>
          ))}
        </div>

        {/* Related memories */}
        {memory.neighbors && memory.neighbors.length > 0 && (
          <div>
            <div className="text-[8px] uppercase tracking-[0.25em] text-emerald-600/60 mb-2">
              // Related
            </div>
            <div className="space-y-1">
              {memory.neighbors.slice(0, 6).map((n) => (
                <div
                  key={n.id}
                  className="flex items-start gap-2 py-1 px-2 rounded-sm"
                  style={{ background: "rgba(0,255,136,0.03)", border: "1px solid rgba(0,255,136,0.07)" }}
                >
                  <span className="shrink-0 mt-1 w-1 h-1 rounded-full bg-emerald-400/40" />
                  <span className="text-[9px] text-neutral-500 leading-snug line-clamp-2">{n.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Main Page ──────────────────────────────────────────────────────
export function MemoriesPage() {
  const [searchParams, setSearchParams]     = useSearchParams()
  const [memories, setMemories]             = useState<Memory[]>([])
  const [domains, setDomains]               = useState<Domain[]>([])
  const [selectedMemory, setSelectedMemory] = useState<MemoryDetail | null>(null)
  const [loading, setLoading]               = useState(true)
  const [loadingMore, setLoadingMore]       = useState(false)
  const [search, setSearch]                 = useState("")
  const [activeDomain, setActiveDomain]     = useState<string | null>(() => searchParams.get("domain"))
  const [activeSubdomain, setActiveSubdomain] = useState<string | null>(null)
  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set())
  const [activeTier, setActiveTier]         = useState<number | null>(null)
  const [sort, setSort]                     = useState("newest")
  const [page, setPage]                     = useState(0)
  const [hasMore, setHasMore]               = useState(true)
  const [showAddModal, setShowAddModal]     = useState(false)
  const [selected, setSelected]             = useState<Set<string>>(new Set())
  const [selectMode, setSelectMode]         = useState(false)
  const [typewriterText, setTypewriterText] = useState("")
  const [bulkDeleting, setBulkDeleting]     = useState(false)

  useEffect(() => {
    api.domains().then(setDomains).catch(() => {})
    const domainParam = searchParams.get("domain")
    if (domainParam) {
      setExpandedDomains(new Set([domainParam]))
      setSearchParams({}, { replace: true })
    }
  }, [])

  const loadMemories = useCallback(
    (reset = false) => {
      const pageNum = reset ? 0 : page
      const params: Record<string, any> = {}
      if (activeDomain) params.domain = activeDomain
      if (activeSubdomain) params.subdomain = activeSubdomain
      if (search) params.q = search
      if (activeTier !== null) params.tier = activeTier
      const loader = reset ? setLoading : setLoadingMore
      loader(true)
      api.memories({ ...params, limit: 30, offset: pageNum * 30 } as any)
        .then((data) => {
          let sorted = [...data]
          if (sort === "oldest") sorted.sort((a, b) => a.created_at.localeCompare(b.created_at))
          else if (sort === "newest") sorted.sort((a, b) => b.created_at.localeCompare(a.created_at))
          else if (sort === "tier-asc") sorted.sort((a, b) => a.tier - b.tier)
          else if (sort === "tier-desc") sorted.sort((a, b) => b.tier - a.tier)
          reset ? setMemories(sorted) : setMemories((prev) => [...prev, ...sorted])
          setHasMore(data.length === 30)
        })
        .finally(() => loader(false))
    },
    [page, activeDomain, activeSubdomain, search, activeTier, sort]
  )

  useEffect(() => {
    setPage(0)
    loadMemories(true)
  }, [activeDomain, activeSubdomain, search, activeTier, sort])

  useEffect(() => {
    if (!search && !activeDomain) {
      let idx = 0
      const id = setInterval(() => {
        setTypewriterText(TYPEWRITER_TEXT.slice(0, idx + 1))
        if (++idx >= TYPEWRITER_TEXT.length) idx = 0
      }, 100)
      return () => clearInterval(id)
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
    if (selectMode) {
      setSelected((prev) => {
        const next = new Set(prev)
        next.has(id) ? next.delete(id) : next.add(id)
        return next
      })
      return
    }
    try {
      const detail = await api.memory(id)
      setSelectedMemory(detail)
    } catch {}
  }

  const toggleDomain = (name: string) => {
    setExpandedDomains((prev) => {
      const next = new Set(prev)
      next.has(name) ? next.delete(name) : next.add(name)
      return next
    })
  }

  const handleCreated = (mem: Memory) => {
    setMemories((prev) => [mem, ...prev])
    setShowAddModal(false)
    api.domains().then(setDomains).catch(() => {})
  }

  const handleUpdated = (updated: Memory) => {
    setMemories((prev) => prev.map((m) => (m.id === updated.id ? { ...m, text: updated.text } : m)))
    setSelectedMemory((prev) => prev ? { ...prev, text: updated.text } : prev)
  }

  const handleDeleted = (id: string) => {
    setMemories((prev) => prev.filter((m) => m.id !== id))
    setSelectedMemory(null)
    api.domains().then(setDomains).catch(() => {})
  }

  const bulkDelete = async () => {
    if (selected.size === 0) return
    setBulkDeleting(true)
    try {
      await Promise.all([...selected].map((id) => api.deleteMemory(id)))
      setMemories((prev) => prev.filter((m) => !selected.has(m.id)))
      setSelected(new Set())
      setSelectMode(false)
      api.domains().then(setDomains).catch(() => {})
    } finally {
      setBulkDeleting(false)
    }
  }

  const { theme } = useTheme()

  return (
    <div className={`relative flex h-full min-h-0 font-mono ${theme === "dark" ? "bg-[#020d0d]" : "bg-slate-100"}`}>
      {theme === "dark" && <ScanlineOverlay />}

      <div className="pointer-events-none absolute inset-0 z-0"
        style={{ background: theme === "dark" ? "radial-gradient(ellipse at 0% 50%, rgba(0,255,136,0.06), transparent 40%)" : "radial-gradient(ellipse at 0% 50%, rgba(0,196,188,0.03), transparent 40%)" }} />

      {/* ── Domain sidebar ── */}
      <div
        className="relative z-10 w-48 shrink-0 flex flex-col overflow-y-auto"
        style={{ borderRight: "1px solid rgba(0,255,136,0.12)", background: "rgba(0,5,16,0.8)" }}
      >
        <div className="px-4 py-3" style={{ borderBottom: "1px solid rgba(0,255,136,0.1)" }}>
          <div className="text-[8px] uppercase tracking-[0.25em] text-emerald-600/60 mb-0.5">// Filter</div>
          <div className="text-[11px] uppercase tracking-wider text-neutral-400">Domains</div>
        </div>

        <div className="flex-1 px-2 py-2 space-y-0.5">
          {domains.map((domain, idx) => {
            const color = TIER_COLORS[idx % 4]
            const isActive = activeDomain === domain.name
            const isExpanded = expandedDomains.has(domain.name)
            return (
              <div key={domain.name}>
                <button
                  onClick={() => {
                    setActiveDomain(isActive ? null : domain.name)
                    if (isActive) setActiveSubdomain(null)
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
                  <span className="flex-1 truncate">{domain.name}</span>
                  <span className="text-[9px] font-mono px-1 py-0.5 rounded-sm"
                    style={{ color, background: `${color}18` }}>{domain.total}</span>
                </button>

                <div className={`ml-3 overflow-hidden transition-all ${isExpanded || isActive ? "max-h-40" : "max-h-0"}`}>
                  {domain.subdomains?.map((sub) => {
                    const isSubActive = activeSubdomain === sub.name
                    return (
                      <button
                        key={sub.name}
                        onClick={() => { setActiveDomain(domain.name); setActiveSubdomain(sub.name) }}
                        className="w-full text-left px-2 py-1 text-[9px] transition-colors flex items-center gap-1.5"
                        style={{ color: isSubActive ? color : "#525252" }}
                      >
                        <div className="h-px w-2.5 shrink-0" style={{ background: `${color}30` }} />
                        <span className="truncate">{sub.name}</span>
                        <span className="ml-auto">{sub.count}</span>
                      </button>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>

        {(activeDomain || activeSubdomain) && (
          <div className="px-3 py-2.5" style={{ borderTop: "1px solid rgba(0,255,136,0.08)" }}>
            <button
              onClick={() => { setActiveDomain(null); setActiveSubdomain(null) }}
              className="flex items-center gap-1.5 text-[9px] uppercase tracking-widest text-neutral-600 hover:text-emerald-400 transition-colors"
            >
              <X className="w-3 h-3" /> clear filter
            </button>
          </div>
        )}
      </div>

      {/* ── Main content ── */}
      <div className="relative z-10 flex-1 flex flex-col min-w-0 px-5 py-4">

        {/* Top bar: search + add button */}
        <div className="flex items-center gap-3 mb-3">
          <div className="relative flex-1">
            <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-emerald-400/40"
              style={{ boxShadow: "0 0 6px rgba(0,255,136,0.4)" }} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={typewriterText || TYPEWRITER_TEXT}
              className="w-full bg-transparent py-2.5 pl-4 pr-4 text-[12px] text-neutral-100 placeholder:text-neutral-700 focus:outline-none"
              style={{ borderBottom: "1px solid rgba(0,255,136,0.2)" }}
              onFocus={(e) => (e.target.style.borderBottomColor = "rgba(0,255,136,0.5)")}
              onBlur={(e) => (e.target.style.borderBottomColor = "rgba(0,255,136,0.2)")}
            />
          </div>

          {/* Add button */}
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-1.5 px-3 py-2 text-[10px] uppercase tracking-wider transition-all hover:bg-emerald-400/10"
            style={{ border: "1px solid rgba(0,255,136,0.3)", color: "#00ff88" }}
          >
            <Plus className="w-3.5 h-3.5" />
            Add
          </button>
        </div>

        {/* Filter bar: tier + sort + select mode */}
        <div className="flex items-center gap-3 mb-4 flex-wrap">
          {/* Tier filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-neutral-600 uppercase tracking-wider">Tier:</span>
            <div className="flex gap-1">
              {TIER_LABELS.map(({ tier, label }) => {
                const isActive = activeTier === tier
                const color = getTierColor(tier)
                return (
                  <button
                    key={tier}
                    onClick={() => setActiveTier(isActive ? null : tier)}
                    className="px-2 py-1 text-[9px] uppercase tracking-wider rounded-sm transition-all"
                    style={{
                      color: isActive ? color : "#525252",
                      background: isActive ? `${color}15` : "transparent",
                      border: `1px solid ${isActive ? color : "rgba(255,255,255,0.08)"}`,
                    }}
                  >
                    {label}
                  </button>
                )
              })}
              {activeTier !== null && (
                <button onClick={() => setActiveTier(null)}
                  className="px-1.5 text-[10px] text-neutral-500 hover:text-neutral-300">×</button>
              )}
            </div>
          </div>

          {/* Sort */}
          <div className="flex items-center gap-1.5 ml-auto">
            <ArrowUpDown className="w-3 h-3 text-neutral-600" />
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value)}
              className="bg-transparent text-[9px] text-neutral-500 uppercase tracking-wider outline-none cursor-pointer hover:text-neutral-300 transition-colors"
            >
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value} className="bg-[#020d0d] normal-case">
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          {/* Select mode toggle */}
          <button
            onClick={() => { setSelectMode(!selectMode); setSelected(new Set()) }}
            className="text-[9px] uppercase tracking-wider px-2 py-1 transition-all"
            style={{
              color: selectMode ? "#00ff88" : "#525252",
              border: `1px solid ${selectMode ? "rgba(0,255,136,0.3)" : "rgba(255,255,255,0.08)"}`,
            }}
          >
            {selectMode ? "cancel select" : "select"}
          </button>
        </div>

        {/* Bulk action bar */}
        {selectMode && selected.size > 0 && (
          <div className="flex items-center gap-3 mb-3 px-3 py-2 rounded-sm"
            style={{ background: "rgba(239,68,68,0.06)", border: "1px solid rgba(239,68,68,0.2)" }}>
            <span className="text-[10px] text-neutral-400">{selected.size} selected</span>
            <button
              onClick={bulkDelete}
              disabled={bulkDeleting}
              className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-red-400 hover:text-red-300 transition-colors disabled:opacity-40 ml-auto"
            >
              <Trash2 className="w-3 h-3" />
              {bulkDeleting ? "deleting..." : `delete ${selected.size}`}
            </button>
            <button
              onClick={() => setSelected(new Set())}
              className="text-[9px] text-neutral-600 hover:text-neutral-400 transition-colors"
            >
              clear
            </button>
          </div>
        )}

        {/* Count */}
        {(activeDomain || search || activeTier !== null) && !loading && (
          <div className="text-[9px] text-emerald-500/60 mb-2">
            {memories.length} {memories.length === 1 ? "memory" : "memories"} found
          </div>
        )}

        {/* Memory grid */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="text-[10px] uppercase tracking-widest text-emerald-500/50 animate-pulse py-12 text-center">
              syncing memory graph...
            </div>
          ) : memories.length === 0 ? (
            <div className="text-center py-16">
              <div className="text-[10px] text-neutral-700 uppercase tracking-widest mb-2">no memories found</div>
              <button onClick={() => setShowAddModal(true)}
                className="text-[10px] text-emerald-500/60 hover:text-emerald-400 transition-colors uppercase tracking-wider">
                + add the first one
              </button>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2">
                {memories.map((mem) => {
                  const color = getTierColor(mem.tier)
                  const isSelected = selected.has(mem.id)
                  const isActive = selectedMemory?.id === mem.id
                  return (
                    <button
                      key={mem.id}
                      onClick={() => selectMemory(mem.id)}
                      className="group text-left rounded-sm overflow-hidden transition-all duration-150"
                      style={{
                        background: isSelected
                          ? "rgba(239,68,68,0.06)"
                          : isActive
                          ? `${color}08`
                          : "rgba(0,5,16,0.7)",
                        border: isSelected
                          ? "1px solid rgba(239,68,68,0.3)"
                          : isActive
                          ? `1px solid ${color}40`
                          : "1px solid rgba(255,255,255,0.05)",
                        borderLeft: `2px solid ${isSelected ? "#ef4444" : color}`,
                      }}
                    >
                      <div className="px-3 pt-2.5 pb-2">
                        <div className="flex items-center gap-2 mb-1.5">
                          {selectMode && (
                            <div
                              className="w-3.5 h-3.5 rounded-sm shrink-0 border flex items-center justify-center"
                              style={{
                                borderColor: isSelected ? "#ef4444" : "rgba(255,255,255,0.2)",
                                background: isSelected ? "rgba(239,68,68,0.2)" : "transparent",
                              }}
                            >
                              {isSelected && <Check className="w-2.5 h-2.5 text-red-400" />}
                            </div>
                          )}
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
                      <div className="px-3 py-1.5 flex items-center justify-between"
                        style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                        <span className="text-[8px] text-neutral-700 font-mono">{mem.created_at?.slice(0, 10) || "—"}</span>
                        {!selectMode && (
                          <span className="text-[8px] uppercase tracking-wider text-emerald-500/0 group-hover:text-emerald-500/70 transition-colors">
                            view →
                          </span>
                        )}
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
                    borderColor: "rgba(0,255,136,0.2)",
                    color: loadingMore ? "rgba(0,255,136,0.3)" : "rgba(0,255,136,0.6)",
                  }}
                >
                  {loadingMore ? "loading..." : "load more"}
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* ── Detail panel ── */}
      {selectedMemory && (
        <DetailPanel
          memory={selectedMemory}
          onClose={() => setSelectedMemory(null)}
          onUpdated={handleUpdated}
          onDeleted={handleDeleted}
        />
      )}

      {/* ── Add modal ── */}
      {showAddModal && (
        <AddMemoryModal
          onClose={() => setShowAddModal(false)}
          onCreated={handleCreated}
        />
      )}
    </div>
  )
}
