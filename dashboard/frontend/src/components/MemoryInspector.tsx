import { useEffect, useState } from "react"
import { api, type MemoryDetail } from "@/lib/api"
import { HudPanel, StatusDot } from "@/components/hud"

const TIER_COLORS = ["#00ff88", "#00e5ff", "#aaff00", "#00ff66"]
const TIER_NAMES = ["Core Context", "Anchor", "Leaf", "Procedural"]

interface Props {
  memoryId: string | null
  onClose: () => void
  onDeleted: (id: string) => void
}

export function MemoryInspector({ memoryId, onClose, onDeleted }: Props) {
  const [data, setData] = useState<MemoryDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!memoryId) { setData(null); return }
    setLoading(true)
    setError(null)
    setEditing(false)
    api.memory(memoryId)
      .then((d) => { setData(d); setDraft(d.text) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [memoryId])

  if (!memoryId) return null

  const save = async () => {
    if (!data) return
    setSaving(true)
    try {
      const updated = await api.updateMemory(data.id, draft)
      setData({ ...data, text: updated.text })
      setEditing(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : "save failed")
    } finally {
      setSaving(false)
    }
  }

  const remove = async () => {
    if (!data) return
    if (!confirm("delete this memory?")) return
    try {
      await api.deleteMemory(data.id)
      onDeleted(data.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : "delete failed")
    }
  }

  const tierIdx = data ? Math.max(0, Math.min(3, data.tier - 1)) : 0

  return (
    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-30 w-[520px] max-w-[92vw] pointer-events-auto">
      <HudPanel id={`MEM.${memoryId.slice(0, 6).toUpperCase()}`} className="hud-panel" progressValue={loading ? 30 : 100}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{
                backgroundColor: TIER_COLORS[tierIdx],
                boxShadow: `0 0 8px ${TIER_COLORS[tierIdx]}`,
              }}
            />
            <span className="text-[10px] uppercase tracking-widest text-neutral-300">
              {data?.tier_name || TIER_NAMES[tierIdx]}
            </span>
            <StatusDot label="LOADED" color="#22d3ee" />
          </div>
          <button
            onClick={onClose}
            className="border border-neutral-700 px-2 py-0.5 text-[11px] text-neutral-500 hover:text-neutral-200 hover:border-neutral-500 transition"
          >
            ×
          </button>
        </div>

        {loading && <div className="text-[11px] text-neutral-500">resolving node...</div>}
        {error && <div className="text-[11px] text-red-400/80 mb-2">{error}</div>}

        {data && (
          <>
            <div className="mb-3">
              <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-1">content</div>
              {editing ? (
                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  rows={4}
                  className="w-full bg-black/40 border border-emerald-400/30 focus:border-emerald-400/70 outline-none text-[12px] text-neutral-100 p-2 font-mono"
                />
              ) : (
                <div className="text-[12px] text-neutral-100 leading-relaxed whitespace-pre-wrap">
                  {data.text}
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2 mb-3 text-[10px]">
              <Field label="domain" value={data.domain} />
              <Field label="subdomain" value={data.subdomain} />
              <Field label="created" value={new Date(data.created_at).toLocaleString()} />
              <Field label="node id" value={data.id.slice(0, 12) + "..."} mono />
            </div>

            <div className="mb-3">
              <div className="text-[9px] uppercase tracking-wider text-neutral-500 mb-1.5">
                neighbors ({data.neighbors.length})
              </div>
              <div className="space-y-1 max-h-[140px] overflow-y-auto pr-1">
                {data.neighbors.length === 0 && (
                  <div className="text-[11px] text-neutral-600 italic">no connections</div>
                )}
                {data.neighbors.map((n) => (
                  <div
                    key={n.id}
                    className="flex items-start gap-2 p-1.5 border-l-2 border-emerald-400/20 hover:border-emerald-400/60 hover:bg-emerald-400/5 transition"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-[11px] text-neutral-200 line-clamp-1">{n.text}</div>
                      <div className="flex items-center gap-2 text-[9px] text-neutral-500 uppercase tracking-wider mt-0.5">
                        {n.relation && <span>{n.relation}</span>}
                        {n.weight !== undefined && (
                          <span className="text-emerald-400/70 tabular-nums">w {n.weight.toFixed(2)}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2 pt-2 border-t border-emerald-400/10">
              {editing ? (
                <>
                  <button
                    onClick={save}
                    disabled={saving || !draft.trim()}
                    className="scan-button border border-emerald-400/40 px-3 py-1 text-[10px] text-emerald-400/80 uppercase tracking-wider"
                  >
                    {saving ? "saving..." : "[ save ]"}
                  </button>
                  <button
                    onClick={() => { setEditing(false); setDraft(data.text) }}
                    className="border border-neutral-700 px-3 py-1 text-[10px] text-neutral-500 hover:text-neutral-300 uppercase tracking-wider transition"
                  >
                    cancel
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => setEditing(true)}
                    className="scan-button border border-emerald-400/40 px-3 py-1 text-[10px] text-emerald-400/70 uppercase tracking-wider"
                  >
                    [ edit ]
                  </button>
                  <button
                    onClick={remove}
                    className="border border-red-400/40 px-3 py-1 text-[10px] text-red-400/70 hover:text-red-400 hover:border-red-400/70 uppercase tracking-wider transition"
                  >
                    [ delete ]
                  </button>
                </>
              )}
            </div>
          </>
        )}
      </HudPanel>
    </div>
  )
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="border border-emerald-400/10 bg-black/30 p-1.5">
      <div className="text-[8px] uppercase tracking-widest text-neutral-500 mb-0.5">{label}</div>
      <div className={`text-[11px] text-neutral-200 truncate ${mono ? "font-mono" : ""}`}>{value}</div>
    </div>
  )
}
