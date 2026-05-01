import { useState } from "react"
import { api, type Memory } from "@/lib/api"
import { HudPanel } from "@/components/hud"

const TIER_OPTIONS = [
  { value: 3, label: "Leaf — atomic fact / observation" },
  { value: 2, label: "Anchor — stable reference" },
  { value: 1, label: "Context — core semantic hub" },
  { value: 4, label: "Procedural — skill / workflow" },
]

const TIER_COLORS = ["#00ff88", "#ff8c26", "#ffd700", "#ff4d6d"]

interface Props {
  onCreated?: (memory: Memory) => void
}

export function MemoryCreator({ onCreated }: Props) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState("")
  const [tier, setTier] = useState(3)
  const [domain, setDomain] = useState("")
  const [subdomain, setSubdomain] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successId, setSuccessId] = useState<string | null>(null)

  const tierIdx = Math.max(0, Math.min(3, tier - 1))
  const accentColor = TIER_COLORS[tierIdx]

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setSuccessId(null)
    try {
      const mem = await api.createMemory({
        text: text.trim(),
        tier,
        domain: domain.trim() || undefined,
        subdomain: subdomain.trim() || undefined,
      })
      setSuccessId(mem.id.slice(0, 8))
      setText("")
      setDomain("")
      setSubdomain("")
      onCreated?.(mem)
      setTimeout(() => { setSuccessId(null); setOpen(false) }, 2000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "injection failed")
    } finally {
      setLoading(false)
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="pointer-events-auto border bg-transparent px-3 py-1.5 text-[10px] font-mono uppercase tracking-widest transition-all duration-200"
        style={{
          borderColor: "rgba(0, 160, 80,0.4)",
          color: "rgba(0, 160, 80,0.65)",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = "rgba(0, 160, 80,0.8)"
          e.currentTarget.style.color = "rgba(0, 160, 80,0.9)"
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = "rgba(0, 160, 80,0.4)"
          e.currentTarget.style.color = "rgba(0, 160, 80,0.65)"
        }}
      >
        [ + inject memory ]
      </button>
    )
  }

  return (
    <div className="pointer-events-auto w-[360px]">
      <HudPanel id="SYS.INJ.05" className="hud-panel" progressValue={loading ? 45 : successId ? 100 : 0}>
        <div className="flex items-center justify-between mb-3">
          <div className="text-[10px] uppercase tracking-widest text-neutral-400">
            Memory Injection
          </div>
          <button
            onClick={() => { setOpen(false); setError(null); setSuccessId(null) }}
            className="border border-neutral-700 px-2 py-0.5 text-[11px] text-neutral-500 hover:text-neutral-200 hover:border-neutral-500 transition"
          >
            ×
          </button>
        </div>

        {successId ? (
          <div className="py-3 text-center">
            <div className="text-[11px] text-purple-400 mb-1">node injected</div>
            <div className="text-[10px] text-neutral-500 font-mono">{successId}...</div>
          </div>
        ) : (
          <form onSubmit={submit} className="space-y-3">
            <div>
              <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-1">content</div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="enter memory text..."
                rows={3}
                disabled={loading}
                className="w-full bg-black/40 border border-emerald-400/20 focus:border-emerald-400/50 outline-none text-[12px] text-neutral-100 p-2 font-mono placeholder:text-neutral-600 resize-none transition-colors"
              />
            </div>

            <div>
              <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-1">tier</div>
              <div className="grid grid-cols-2 gap-1">
                {TIER_OPTIONS.map((opt) => {
                  const idx = Math.max(0, Math.min(3, opt.value - 1))
                  const active = tier === opt.value
                  return (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setTier(opt.value)}
                      disabled={loading}
                      className="text-left px-2 py-1.5 border text-[10px] transition-all"
                      style={{
                        borderColor: active ? TIER_COLORS[idx] + "80" : "rgba(255,255,255,0.08)",
                        backgroundColor: active ? TIER_COLORS[idx] + "15" : "transparent",
                        color: active ? TIER_COLORS[idx] : "rgba(163,163,163,0.7)",
                      }}
                    >
                      <span className="font-medium">{opt.label.split(" — ")[0]}</span>
                      <span className="block text-[9px] opacity-60 mt-0.5 leading-tight">
                        {opt.label.split(" — ")[1]}
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-1">
                  domain <span className="text-neutral-600">(optional)</span>
                </div>
                <input
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  placeholder="auto-classify"
                  disabled={loading}
                  className="w-full bg-black/40 border border-emerald-400/20 focus:border-emerald-400/50 outline-none text-[11px] text-neutral-100 px-2 py-1.5 font-mono placeholder:text-neutral-600 transition-colors"
                />
              </div>
              <div>
                <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-1">
                  subdomain <span className="text-neutral-600">(optional)</span>
                </div>
                <input
                  value={subdomain}
                  onChange={(e) => setSubdomain(e.target.value)}
                  placeholder="auto-classify"
                  disabled={loading}
                  className="w-full bg-black/40 border border-emerald-400/20 focus:border-emerald-400/50 outline-none text-[11px] text-neutral-100 px-2 py-1.5 font-mono placeholder:text-neutral-600 transition-colors"
                />
              </div>
            </div>

            {error && (
              <div className="text-[11px] text-red-400/80">{error}</div>
            )}

            <div className="flex items-center gap-2 pt-1 border-t border-white/5">
              <button
                type="submit"
                disabled={loading || !text.trim()}
                className="scan-button border px-4 py-1.5 text-[10px] uppercase tracking-wider transition-all disabled:opacity-40"
                style={{
                  borderColor: accentColor + "70",
                  color: accentColor + "cc",
                }}
              >
                {loading ? "injecting..." : "[ inject ]"}
              </button>
              <div className="text-[9px] text-neutral-600 uppercase tracking-wider">
                domain auto-classified when blank
              </div>
            </div>
          </form>
        )}
      </HudPanel>
    </div>
  )
}
