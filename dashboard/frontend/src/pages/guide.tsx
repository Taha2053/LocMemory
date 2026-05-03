import { useState } from "react"
import { ScanlineOverlay } from "@/components/hud"

interface Section {
  id: string
  tag: string
  title: string
  content: React.ReactNode
}

function TierCard({ tier, name, color, desc, example }: {
  tier: number; name: string; color: string; desc: string; example: string
}) {
  return (
    <div
      className="rounded-sm px-4 py-3"
      style={{ background: "rgba(0,5,16,0.7)", border: `1px solid ${color}20`, borderLeftWidth: 3, borderLeftColor: color, borderLeftStyle: "solid" }}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[9px] font-mono text-neutral-600">T{tier}</span>
        <span className="text-[11px] uppercase tracking-widest font-semibold" style={{ color }}>{name}</span>
      </div>
      <p className="text-[11px] text-neutral-400 mb-2">{desc}</p>
      <div className="text-[9px] text-neutral-600 italic">e.g. "{example}"</div>
    </div>
  )
}

function StepCard({ step, title, desc }: { step: string; title: string; desc: string }) {
  return (
    <div className="flex gap-4">
      <div
        className="shrink-0 h-7 w-7 rounded-sm flex items-center justify-center text-[10px] font-mono font-bold"
        style={{ background: "rgba(0, 196, 188,0.12)", border: "1px solid rgba(0, 196, 188,0.3)", color: "#00c4bc" }}
      >
        {step}
      </div>
      <div>
        <div className="text-[11px] uppercase tracking-wider text-neutral-300 font-semibold mb-0.5">{title}</div>
        <p className="text-[11px] text-neutral-500">{desc}</p>
      </div>
    </div>
  )
}

const SECTIONS: Section[] = [
  {
    id: "what",
    tag: "01",
    title: "What is LocMemory?",
    content: (
      <div className="space-y-3">
        <p className="text-[12px] text-neutral-400 leading-relaxed">
          LocMemory is a <span className="text-emerald-300">local cognitive memory system</span> that runs entirely on your machine.
          It listens to your conversations, extracts meaningful facts, and organises them into a
          graph — so when you ask a question, it retrieves the most relevant memories to give your
          LLM useful context.
        </p>
        <p className="text-[12px] text-neutral-500 leading-relaxed">
          Think of it as long-term memory for your AI assistant. Every conversation teaches it
          something new. Over time it learns patterns, links related ideas, and surfaces the right
          information at the right moment.
        </p>
        <div className="grid grid-cols-3 gap-3 mt-4">
          {[
            { label: "Local-first", desc: "All data stays on your machine. No cloud, no tracking." },
            { label: "Graph-based", desc: "Memories are nodes. Relationships are edges. Context flows through the graph." },
            { label: "Self-improving", desc: "Hebbian learning and an RL agent tune retrieval quality over time." },
          ].map(({ label, desc }) => (
            <div key={label} className="rounded-sm px-3 py-3"
              style={{ background: "rgba(0, 196, 188,0.04)", border: "1px solid rgba(0, 196, 188,0.12)" }}>
              <div className="text-[10px] uppercase tracking-widest text-emerald-400/80 mb-1">{label}</div>
              <p className="text-[10px] text-neutral-500">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    id: "tiers",
    tag: "02",
    title: "Memory Tiers",
    content: (
      <div className="space-y-3">
        <p className="text-[12px] text-neutral-500 leading-relaxed mb-4">
          Every memory node belongs to one of four tiers. Tiers represent the level of abstraction —
          raw facts at the bottom, synthesised knowledge at the top.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <TierCard tier={1} name="Context" color="#00c4bc"
            desc="Recent, session-level facts. High recency weight. Fade over time."
            example="I'm currently debugging the auth module" />
          <TierCard tier={2} name="Anchor" color="#ff8c26"
            desc="Consolidated summaries created automatically by the Louvain clustering algorithm."
            example="You frequently work on backend Python projects" />
          <TierCard tier={3} name="Leaf" color="#aaff00"
            desc="Stable long-term facts extracted from your conversations."
            example="I run three times a week" />
          <TierCard tier={4} name="Procedural" color="#c8a8ff"
            desc="Cross-domain patterns detected across multiple memory clusters."
            example="When starting a new project, you tend to begin with the data layer" />
        </div>
      </div>
    ),
  },
  {
    id: "retrieval",
    tag: "03",
    title: "How Retrieval Works",
    content: (
      <div className="space-y-4">
        <p className="text-[12px] text-neutral-500 leading-relaxed">
          When you send a query, the system scores every candidate memory and returns the top results.
          The score has three components:
        </p>
        <div className="space-y-2">
          {[
            { label: "Cosine similarity", weight: "60%", color: "#00c4bc", desc: "How semantically close is the memory text to your query? Computed via all-MiniLM-L6-v2 embeddings." },
            { label: "Recency", weight: "20%", color: "#ff8c26", desc: "More recent memories score higher. Context-tier memories decay faster than Leaf-tier." },
            { label: "Category match", weight: "20%", color: "#ffd700", desc: "Does the memory's domain match the detected domain of your query?" },
          ].map(({ label, weight, color, desc }) => (
            <div key={label} className="flex gap-4 rounded-sm px-4 py-3"
              style={{ background: "rgba(0,5,16,0.6)", border: `1px solid ${color}20` }}>
              <div className="shrink-0 text-center">
                <div className="text-[18px] font-bold font-mono" style={{ color }}>{weight}</div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider mb-0.5" style={{ color }}>{label}</div>
                <p className="text-[11px] text-neutral-500">{desc}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-neutral-600 mt-2">
          When the RL agent is trained, it re-ranks the candidate pool using a PPO model that
          optimises for semantic overlap, diversity, and token efficiency.
        </p>
      </div>
    ),
  },
  {
    id: "hebbian",
    tag: "04",
    title: "Hebbian Learning",
    content: (
      <div className="space-y-3">
        <p className="text-[12px] text-neutral-500 leading-relaxed">
          Inspired by neuroscience: <em className="text-neutral-300">"neurons that fire together, wire together."</em>{" "}
          Every time two memories are retrieved in the same query, the edge between them gets stronger.
          Edges decay slowly over time if not reinforced.
        </p>
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-sm px-4 py-3" style={{ background: "rgba(0,5,16,0.6)", border: "1px solid rgba(255,140,38,0.15)" }}>
            <div className="text-[9px] uppercase tracking-widest text-orange-400/70 mb-1">Strengthening</div>
            <p className="text-[11px] text-neutral-500">Co-retrieved memories get their edge weight increased by the learning rate (default 0.2).</p>
          </div>
          <div className="rounded-sm px-4 py-3" style={{ background: "rgba(0,5,16,0.6)", border: "1px solid rgba(0, 196, 188,0.15)" }}>
            <div className="text-[9px] uppercase tracking-widest text-emerald-400/70 mb-1">Decay</div>
            <p className="text-[11px] text-neutral-500">Unused edges lose weight over time (λ = 0.01). Edges below 0.01 are pruned.</p>
          </div>
        </div>
        <p className="text-[11px] text-neutral-600">
          You can view edge weight distribution and trigger manual decay from the <span className="text-emerald-400/60">Graph</span> page.
        </p>
      </div>
    ),
  },
  {
    id: "domains",
    tag: "05",
    title: "Domains & Subdomains",
    content: (
      <div className="space-y-3">
        <p className="text-[12px] text-neutral-500 leading-relaxed">
          Every memory is automatically classified into a domain and subdomain using the embedding
          model. Classification helps retrieval by boosting memories that match the topic of your query.
        </p>
        <div className="grid grid-cols-2 gap-2">
          {[
            { domain: "health", subs: ["fitness", "nutrition", "mental", "sleep"] },
            { domain: "programming", subs: ["python", "javascript", "algorithms", "devops"] },
            { domain: "work", subs: ["meetings", "deadlines", "projects", "team"] },
            { domain: "learning", subs: ["books", "courses", "research", "notes"] },
            { domain: "personal", subs: ["family", "hobbies", "goals", "habits"] },
            { domain: "finance", subs: ["budget", "investments", "expenses", "savings"] },
          ].map(({ domain, subs }) => (
            <div key={domain} className="rounded-sm px-3 py-2.5"
              style={{ background: "rgba(0,5,16,0.6)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <div className="text-[10px] uppercase tracking-widest text-emerald-400/70 mb-1.5">{domain}</div>
              <div className="flex flex-wrap gap-1">
                {subs.map(s => (
                  <span key={s} className="text-[8px] px-1.5 py-0.5 rounded-sm text-neutral-600"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    {s}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    ),
  },
  {
    id: "dashboard",
    tag: "06",
    title: "Dashboard Pages",
    content: (
      <div className="space-y-2.5">
        {[
          { page: "Graph", color: "#00c4bc", desc: "3D force-directed visualisation of the memory graph. Nodes are memories, edges are relationships. Use it to explore how your knowledge is connected. Trigger consolidation and Hebbian decay from here." },
          { page: "Memories", color: "#00ff88", desc: "Browse, search and filter all stored memories. Filter by domain, subdomain, or tier. Click any memory to inspect its neighbors and metadata." },
          { page: "Domains", color: "#aaff00", desc: "Aggregated view of your knowledge by domain. See which topics dominate your memory graph and drill into subdomains." },
          { page: "Retrieval", color: "#ff8c26", desc: "Test retrieval directly. Enter a query and see ranked results with their scores. Compare hybrid vs RL ranking side by side." },
          { page: "Metrics", color: "#ffd700", desc: "Quality analytics for recent retrievals. Track average score, latency, keyword overlap, and Precision@5 over time." },
          { page: "Settings", color: "#c8a8ff", desc: "Configure every aspect of the system — LLM provider, retrieval weights, Hebbian parameters, RL agent, security and more." },
        ].map(({ page, color, desc }) => (
          <div key={page} className="flex gap-4 rounded-sm px-4 py-3"
            style={{ background: "rgba(0,5,16,0.6)", borderLeft: `2px solid ${color}` }}>
            <div className="shrink-0 w-20">
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color }}>{page}</span>
            </div>
            <p className="text-[11px] text-neutral-500">{desc}</p>
          </div>
        ))}
      </div>
    ),
  },
  {
    id: "quickstart",
    tag: "07",
    title: "Quick Start",
    content: (
      <div className="space-y-4">
        <p className="text-[12px] text-neutral-500">Get up and running in three steps:</p>
        <div className="space-y-4">
          <StepCard step="1" title="Start the backend"
            desc="Run `uv run uvicorn dashboard.backend.main:app --reload --port 8000` from the LocMemory directory." />
          <StepCard step="2" title="Start the chat interface"
            desc="Run `uv run python -m core.chat` to open the TUI. Type normally — LocMemory will extract memories from every exchange in the background." />
          <StepCard step="3" title="Explore this dashboard"
            desc="Open localhost:5173. Use the Memories page to see what was learned, the Retrieval page to test queries, and the Graph page to visualise connections." />
        </div>
        <div className="mt-4 rounded-sm px-4 py-3"
          style={{ background: "rgba(0, 196, 188,0.04)", border: "1px solid rgba(0, 196, 188,0.15)" }}>
          <div className="text-[9px] uppercase tracking-widest text-emerald-400/70 mb-1.5">TUI Commands</div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1">
            {[
              ["/activate", "enable memory extraction"],
              ["/deactivate", "pause memory extraction"],
              ["/mem", "show memory statistics"],
              ["/stats", "show graph statistics"],
              ["/list [domain]", "list stored memories"],
              ["/help", "show all commands"],
            ].map(([cmd, desc]) => (
              <div key={cmd} className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-emerald-400/60 shrink-0">{cmd}</span>
                <span className="text-[9px] text-neutral-600">{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    ),
  },
]

export function GuidePage() {
  const [activeSection, setActiveSection] = useState<string>("what")

  const active = SECTIONS.find(s => s.id === activeSection)

  return (
    <div className="relative flex h-full min-h-0 font-mono bg-[#020d0d]">
      <ScanlineOverlay />

      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 10% 30%, rgba(0, 196, 188,0.05), transparent 50%)" }} />

      {/* Left nav */}
      <div
        className="relative z-10 w-52 shrink-0 flex flex-col overflow-y-auto"
        style={{ borderRight: "1px solid rgba(0, 196, 188,0.12)", background: "rgba(0,5,16,0.8)" }}
      >
        <div className="px-4 py-4" style={{ borderBottom: "1px solid rgba(0, 196, 188,0.1)" }}>
          <div className="text-[8px] uppercase tracking-[0.25em] text-emerald-600/50 mb-0.5">// DOC</div>
          <div className="text-[13px] uppercase tracking-widest text-emerald-300 font-semibold"
            style={{ textShadow: "0 0 12px rgba(0, 196, 188,0.4)" }}>
            Guide
          </div>
        </div>

        <nav className="flex-1 px-2 py-3 space-y-0.5">
          {SECTIONS.map(({ id, tag, title }) => {
            const isActive = activeSection === id
            return (
              <button
                key={id}
                onClick={() => setActiveSection(id)}
                className="w-full flex items-center gap-2.5 px-3 py-2 text-left rounded-sm transition-all duration-150"
                style={{
                  background: isActive ? "rgba(0, 196, 188,0.08)" : "transparent",
                  borderLeft: isActive ? "2px solid #00c4bc" : "2px solid transparent",
                  boxShadow: isActive ? "0 0 12px rgba(0, 196, 188,0.05)" : "none",
                }}
              >
                <span className="text-[8px] tabular-nums text-neutral-700 font-mono w-4">{tag}</span>
                <span className={`text-[10px] uppercase tracking-wider ${isActive ? "text-emerald-300" : "text-neutral-500 hover:text-neutral-300"}`}>
                  {title}
                </span>
                {isActive && (
                  <span className="ml-auto h-1 w-1 rounded-full bg-emerald-400 shrink-0"
                    style={{ boxShadow: "0 0 6px rgba(0, 196, 188,0.9)" }} />
                )}
              </button>
            )
          })}
        </nav>

        <div className="px-4 py-3" style={{ borderTop: "1px solid rgba(0, 196, 188,0.08)" }}>
          <div className="text-[8px] text-neutral-700 uppercase tracking-widest">LocMemory v0.1.0</div>
        </div>
      </div>

      {/* Content */}
      <div className="relative z-10 flex-1 overflow-y-auto px-8 py-8">
        {active && (
          <div className="max-w-3xl">
            <div className="mb-6">
              <div className="flex items-center gap-3 mb-1">
                <div className="h-px w-6" style={{ background: "rgba(0, 196, 188,0.4)" }} />
                <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/50">
                  // {active.tag}
                </span>
              </div>
              <h2 className="text-[22px] font-bold tracking-wide text-emerald-300"
                style={{ textShadow: "0 0 20px rgba(0, 196, 188,0.3)" }}>
                {active.title}
              </h2>
            </div>

            <div className="mb-6 h-px w-full"
              style={{ background: "linear-gradient(to right, rgba(0, 196, 188,0.2), transparent)" }} />

            <div>{active.content}</div>

            <div className="flex items-center justify-between mt-10 pt-6"
              style={{ borderTop: "1px solid rgba(0, 196, 188,0.08)" }}>
              {(() => {
                const idx = SECTIONS.findIndex(s => s.id === activeSection)
                const prev = SECTIONS[idx - 1]
                const next = SECTIONS[idx + 1]
                return (
                  <>
                    <button
                      onClick={() => prev && setActiveSection(prev.id)}
                      disabled={!prev}
                      className="text-[10px] uppercase tracking-widest text-neutral-600 hover:text-emerald-400 transition-colors disabled:opacity-20 disabled:cursor-default"
                    >
                      ← {prev?.title || ""}
                    </button>
                    <button
                      onClick={() => next && setActiveSection(next.id)}
                      disabled={!next}
                      className="text-[10px] uppercase tracking-widest text-neutral-600 hover:text-emerald-400 transition-colors disabled:opacity-20 disabled:cursor-default"
                    >
                      {next?.title || ""} →
                    </button>
                  </>
                )
              })()}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
