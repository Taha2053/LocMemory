import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { Save, RotateCcw, CheckCircle, AlertCircle } from "lucide-react"
import { useTheme } from "@/lib/theme"

// ── helpers ──────────────────────────────────────────────────────

function Section({ title, tag, children }: { title: string; tag: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-sm border border-emerald-400/10 overflow-hidden"
      style={{ background: "rgba(0,5,16,0.6)" }}
    >
      <div
        className="flex items-center gap-3 px-5 py-3 border-b border-emerald-400/10"
        style={{ background: "rgba(0, 196, 188,0.04)" }}
      >
        <span className="text-[8px] uppercase tracking-[0.25em] text-emerald-600/50">// {tag}</span>
        <div className="h-px flex-1" style={{ background: "linear-gradient(to right, rgba(0, 196, 188,0.15), transparent)" }} />
        <span className="text-[11px] uppercase tracking-widest text-emerald-400/80 font-semibold">{title}</span>
      </div>
      <div className="px-5 py-4 grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
        {children}
      </div>
    </div>
  )
}

function Toggle({ label, desc, value, onChange }: { label: string; desc?: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div>
        <div className="text-[11px] uppercase tracking-wider text-neutral-300">{label}</div>
        {desc && <div className="text-[9px] text-neutral-600 mt-0.5">{desc}</div>}
      </div>
      <button
        onClick={() => onChange(!value)}
        className="relative shrink-0 h-5 w-9 rounded-full transition-all duration-200"
        style={{
          background: value ? "rgba(0, 196, 188, 0.35)" : "rgba(255,255,255,0.06)",
          border: value ? "1px solid rgba(0, 196, 188,0.6)" : "1px solid rgba(255,255,255,0.1)",
          boxShadow: value ? "0 0 8px rgba(0, 196, 188,0.3)" : "none",
        }}
      >
        <span
          className="absolute top-0.5 h-4 w-4 rounded-full transition-all duration-200"
          style={{
            left: value ? "calc(100% - 18px)" : "2px",
            background: value ? "#00c4bc" : "rgba(255,255,255,0.2)",
            boxShadow: value ? "0 0 6px rgba(0, 196, 188,0.7)" : "none",
          }}
        />
      </button>
    </div>
  )
}

function NumberField({ label, desc, value, onChange, min, max, step = 1 }: {
  label: string; desc?: string; value: number; onChange: (v: number) => void
  min?: number; max?: number; step?: number
}) {
  return (
    <div>
      <label className="block text-[11px] uppercase tracking-wider text-neutral-400 mb-1.5">{label}</label>
      {desc && <div className="text-[9px] text-neutral-600 mb-1.5">{desc}</div>}
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full px-3 py-2 text-[12px] font-mono text-emerald-200 rounded-sm border border-emerald-400/15 bg-transparent focus:outline-none focus:border-emerald-400/40 transition-colors"
        style={{ background: "rgba(0, 196, 188,0.03)" }}
      />
    </div>
  )
}

function TextField({ label, desc, value, onChange }: { label: string; desc?: string; value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <label className="block text-[11px] uppercase tracking-wider text-neutral-400 mb-1.5">{label}</label>
      {desc && <div className="text-[9px] text-neutral-600 mb-1.5">{desc}</div>}
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full px-3 py-2 text-[12px] font-mono text-emerald-200 rounded-sm border border-emerald-400/15 bg-transparent focus:outline-none focus:border-emerald-400/40 transition-colors"
        style={{ background: "rgba(0, 196, 188,0.03)" }}
      />
    </div>
  )
}

function SelectField({ label, desc, value, options, onChange }: {
  label: string; desc?: string; value: string; options: string[]; onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="block text-[11px] uppercase tracking-wider text-neutral-400 mb-1.5">{label}</label>
      {desc && <div className="text-[9px] text-neutral-600 mb-1.5">{desc}</div>}
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full px-3 py-2 text-[12px] font-mono text-emerald-200 rounded-sm border border-emerald-400/15 focus:outline-none focus:border-emerald-400/40 transition-colors appearance-none"
        style={{ background: "rgba(0,5,16,0.8)" }}
      >
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  )
}

// ── main page ─────────────────────────────────────────────────────

export function SettingsPage() {
  const { theme } = useTheme()
  const [draft, setDraft] = useState<Record<string, any> | null>(null)
  const [original, setOriginal] = useState<Record<string, any> | null>(null)
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle")
  const [errorMsg, setErrorMsg] = useState("")
  const [isDirty, setIsDirty] = useState(false)

  useEffect(() => {
    api.config().then(c => {
      setDraft(JSON.parse(JSON.stringify(c)))
      setOriginal(JSON.parse(JSON.stringify(c)))
    })
  }, [])

  const set = (section: string, key: string, value: any) => {
    setDraft(prev => {
      if (!prev) return prev
      const next = { ...prev, [section]: { ...prev[section], [key]: value } }
      setIsDirty(true)
      return next
    })
    if (status !== "idle") setStatus("idle")
  }

  const save = async () => {
    if (!draft) return
    setStatus("saving")
    setErrorMsg("")
    try {
      const res = await api.updateConfig(draft)
      const updated = JSON.parse(JSON.stringify(res.data))
      setDraft(updated)
      setOriginal(updated)
      setIsDirty(false)
      setStatus("saved")
      setTimeout(() => setStatus("idle"), 2500)
    } catch (e: any) {
      setStatus("error")
      setErrorMsg(e.message || String(e))
    }
  }

  const reset = () => {
    if (original) {
      setDraft(JSON.parse(JSON.stringify(original)))
      setIsDirty(false)
      setStatus("idle")
      setErrorMsg("")
    }
  }

  const g = (section: string, key: string, fallback: any = "") =>
    draft?.[section]?.[key] ?? fallback

  return (
    <div className={`relative h-full min-h-0 font-mono overflow-y-auto ${theme === "dark" ? "bg-[#020d0d]" : "bg-slate-100"}`}>
      {theme === "dark" && <ScanlineOverlay />}

      {/* Ambient */}
      <div className="pointer-events-none absolute inset-0"
        style={{ background: theme === "dark" ? "radial-gradient(ellipse at 0% 0%, rgba(0, 196, 188,0.06), transparent 40%)" : "radial-gradient(ellipse at 0% 0%, rgba(0, 196, 188,0.03), transparent 40%)" }} />

      {/* Corner brackets */}
      <div className="pointer-events-none absolute top-3 left-3 h-5 w-5 border-t-2 border-l-2 border-emerald-400/30" />
      <div className="pointer-events-none absolute top-3 right-3 h-5 w-5 border-t-2 border-r-2 border-emerald-400/30" />
      <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-emerald-400/30" />
      <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-emerald-400/30" />

      <div className="relative z-10 max-w-4xl mx-auto px-6 py-8 space-y-6">

        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px w-8 bg-emerald-400/40" />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">Settings</span>
          </div>
          <h1 className="text-2xl font-bold tracking-wide text-emerald-300"
            style={{ textShadow: "0 0 20px rgba(0, 196, 188,0.4)" }}>
            Settings
          </h1>
          <p className="mt-1 text-[11px] text-neutral-600 uppercase tracking-wider">
            System configuration — changes are written to config.yaml on save
          </p>
        </div>

        {draft === null ? (
          <div className="text-[10px] uppercase tracking-widest text-emerald-500/50 animate-pulse py-16 text-center">
            LOADING CONFIG...
          </div>
        ) : (
          <>
            {/* LLM */}
            <Section title="Language Model" tag="01">
              <SelectField label="Provider" value={g("models", "llm", {}).provider ?? "ollama"}
                options={["ollama", "anthropic", "huggingface"]}
                onChange={v => set("models", "llm", { ...g("models", "llm", {}), provider: v })} />
              <TextField label="Model name" value={g("models", "llm", {}).model ?? ""}
                onChange={v => set("models", "llm", { ...g("models", "llm", {}), model: v })} />
              <NumberField label="Temperature" value={g("models", "llm", {}).temperature ?? 0.3}
                min={0} max={2} step={0.05}
                onChange={v => set("models", "llm", { ...g("models", "llm", {}), temperature: v })} />
              <NumberField label="Max tokens" value={g("models", "llm", {}).max_tokens ?? 512}
                min={64} max={4096} step={64}
                onChange={v => set("models", "llm", { ...g("models", "llm", {}), max_tokens: v })} />
            </Section>

            {/* Retrieval */}
            <Section title="Retrieval" tag="02">
              <NumberField label="Semantic weight" desc="Cosine similarity contribution (0–1)"
                value={g("retrieval", "semantic_weight")} min={0} max={1} step={0.05}
                onChange={v => set("retrieval", "semantic_weight", v)} />
              <NumberField label="Graph weight" desc="Graph traversal contribution (0–1)"
                value={g("retrieval", "graph_weight")} min={0} max={1} step={0.05}
                onChange={v => set("retrieval", "graph_weight", v)} />
              <NumberField label="Max results"
                value={g("retrieval", "max_results")} min={1} max={100}
                onChange={v => set("retrieval", "max_results", v)} />
              <NumberField label="Min similarity" desc="Reject results below this threshold"
                value={g("retrieval", "min_similarity")} min={0} max={1} step={0.05}
                onChange={v => set("retrieval", "min_similarity", v)} />
              <NumberField label="Traversal depth"
                value={g("retrieval", "traversal_depth")} min={1} max={5}
                onChange={v => set("retrieval", "traversal_depth", v)} />
            </Section>

            {/* Hebbian */}
            <Section title="Hebbian Learning" tag="03">
              <Toggle label="Enabled" value={g("hebbian", "enabled")}
                onChange={v => set("hebbian", "enabled", v)} />
              <NumberField label="Learning rate" value={g("hebbian", "learning_rate")}
                min={0} max={1} step={0.01}
                onChange={v => set("hebbian", "learning_rate", v)} />
              <NumberField label="Decay lambda" value={g("hebbian", "decay_lambda")}
                min={0} max={0.5} step={0.001}
                onChange={v => set("hebbian", "decay_lambda", v)} />
              <NumberField label="Max edge weight" value={g("hebbian", "max_edge_weight")}
                min={1} max={20} step={0.5}
                onChange={v => set("hebbian", "max_edge_weight", v)} />
            </Section>

            {/* Extraction */}
            <Section title="Memory Extraction" tag="04">
              <Toggle label="Background extraction" desc="Extract memories from each chat turn"
                value={g("extraction", "enable_background_extraction")}
                onChange={v => set("extraction", "enable_background_extraction", v)} />
              <NumberField label="Thread pool size" value={g("extraction", "thread_pool_size")}
                min={1} max={8}
                onChange={v => set("extraction", "thread_pool_size", v)} />
              <NumberField label="Min fact length (chars)" value={g("extraction", "min_fact_length")}
                min={5} max={100}
                onChange={v => set("extraction", "min_fact_length", v)} />
              <NumberField label="Max facts per message" value={g("extraction", "max_facts_per_message")}
                min={1} max={20}
                onChange={v => set("extraction", "max_facts_per_message", v)} />
            </Section>

            {/* Consolidation */}
            <Section title="Memory Consolidation" tag="05">
              <Toggle label="Enabled" desc="Cluster nodes into Tier-2 anchors via Louvain"
                value={g("consolidation", "enabled")}
                onChange={v => set("consolidation", "enabled", v)} />
              <NumberField label="Min cluster size" value={g("consolidation", "cluster_min_size")}
                min={3} max={50}
                onChange={v => set("consolidation", "cluster_min_size", v)} />
              <NumberField label="Run every N additions" value={g("consolidation", "run_every_n_additions")}
                min={5} max={200}
                onChange={v => set("consolidation", "run_every_n_additions", v)} />
              <NumberField label="Max clusters per run" value={g("consolidation", "max_clusters_per_run")}
                min={1} max={20}
                onChange={v => set("consolidation", "max_clusters_per_run", v)} />
            </Section>

            {/* RL */}
            <Section title="RL Agent" tag="06">
              <Toggle label="Enabled" desc="Use PPO agent for memory candidate ranking"
                value={g("rl", "enabled")}
                onChange={v => set("rl", "enabled", v)} />
              <NumberField label="Candidate pool size" value={g("rl", "candidate_pool_size")}
                min={5} max={100}
                onChange={v => set("rl", "candidate_pool_size", v)} />
              <NumberField label="Top-k to return" value={g("rl", "top_k")}
                min={1} max={20}
                onChange={v => set("rl", "top_k", v)} />
              <NumberField label="Token budget" value={g("rl", "token_budget")}
                min={64} max={2048} step={64}
                onChange={v => set("rl", "token_budget", v)} />
            </Section>

            {/* Security */}
            <Section title="Security & Privacy" tag="07">
              <Toggle label="PII detection" desc="Flag and optionally encrypt personal data"
                value={g("security", "pii_detection")}
                onChange={v => set("security", "pii_detection", v)} />
              <Toggle label="Encryption at rest" desc="AES-GCM encryption for sensitive memories"
                value={g("security", "encryption_enabled")}
                onChange={v => set("security", "encryption_enabled", v)} />
            </Section>

            {/* Performance & Debug */}
            <Section title="Performance & Debug" tag="08">
              <Toggle label="Enable caching" value={g("performance", "enable_caching")}
                onChange={v => set("performance", "enable_caching", v)} />
              <NumberField label="Embedding batch size" value={g("performance", "embedding_batch_size")}
                min={1} max={128}
                onChange={v => set("performance", "embedding_batch_size", v)} />
              <NumberField label="Max retrieval latency (ms)" value={g("performance", "max_retrieval_latency_ms")}
                min={10} max={5000}
                onChange={v => set("performance", "max_retrieval_latency_ms", v)} />
              <Toggle label="Trace logs" desc="Verbose debug logging"
                value={g("debug", "enable_trace_logs")}
                onChange={v => set("debug", "enable_trace_logs", v)} />
            </Section>

            {/* Action bar */}
            <div
              className="flex items-center justify-between px-5 py-3 rounded-sm border border-emerald-400/10"
              style={{ background: "rgba(0,5,16,0.6)" }}
            >
              <div className="flex items-center gap-2 text-[10px]">
                {status === "saving" && (
                  <><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-emerald-400/70 uppercase tracking-wider">SAVING...</span></>
                )}
                {status === "saved" && (
                  <><CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-green-400/80 uppercase tracking-wider">CONFIG SAVED</span></>
                )}
                {status === "error" && (
                  <><AlertCircle className="w-3.5 h-3.5 text-red-400" />
                    <span className="text-red-400/80 uppercase tracking-wider truncate max-w-xs">{errorMsg}</span></>
                )}
                {status === "idle" && isDirty && (
                  <span className="text-amber-400/70 uppercase tracking-wider">UNSAVED CHANGES</span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={reset}
                  disabled={!isDirty}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] uppercase tracking-widest border border-white/10 text-neutral-500 hover:text-neutral-300 hover:border-white/20 transition-all duration-150 disabled:opacity-30 disabled:cursor-not-allowed rounded-sm"
                >
                  <RotateCcw className="w-3 h-3" />
                  RESET
                </button>
                <button
                  onClick={save}
                  disabled={status === "saving" || !isDirty}
                  className="flex items-center gap-1.5 px-4 py-1.5 text-[10px] uppercase tracking-widest border transition-all duration-150 disabled:opacity-30 disabled:cursor-not-allowed rounded-sm"
                  style={{
                    borderColor: "rgba(0, 196, 188,0.4)",
                    color: "#22d3ee",
                    background: "rgba(0, 196, 188,0.06)",
                    boxShadow: isDirty ? "0 0 12px rgba(0, 196, 188,0.2)" : "none",
                  }}
                >
                  <Save className="w-3 h-3" />
                  SAVE CONFIG
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
