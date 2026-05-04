import { useState, useRef, useEffect } from "react"
import { api, ChatMessage, ChatResponse, RetrievedResult } from "@/lib/api"
import { Send, Brain, Loader2, ChevronDown, ChevronUp, Zap } from "lucide-react"
import { cn } from "@/lib/utils"
import { useTheme } from "@/context/ThemeContext"

const TIER_COLORS: Record<number, string> = {
  1: "rgba(0,196,188,0.8)",
  2: "rgba(255,140,38,0.8)",
  3: "rgba(200,168,255,0.8)",
  4: "rgba(255,77,109,0.8)",
}

const TIER_LABELS: Record<number, string> = {
  1: "CTX",
  2: "ANC",
  3: "LEAF",
  4: "PROC",
}

interface Turn {
  role: "user" | "assistant"
  content: string
  memories?: RetrievedResult[]
  model?: string
  tokens?: { input: number; output: number }
  retrieval_ms?: number
  error?: boolean
}

function MemoryBadge({ m }: { m: RetrievedResult }) {
  const color = TIER_COLORS[m.tier] ?? "rgba(255,255,255,0.5)"
  return (
    <div
      className="flex items-start gap-1.5 py-1 px-2 rounded text-[10px] font-mono"
      style={{ background: "rgba(0,0,0,0.3)", border: `1px solid ${color}22` }}
    >
      <span
        className="shrink-0 mt-0.5 text-[8px] font-bold px-1 rounded"
        style={{ background: `${color}22`, color }}
      >
        {TIER_LABELS[m.tier] ?? "?"}
      </span>
      <span className="text-neutral-400 leading-snug">{m.text}</span>
      <span className="ml-auto shrink-0 text-[9px] tabular-nums" style={{ color }}>
        {m.score.toFixed(2)}
      </span>
    </div>
  )
}

function AssistantBubble({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState(false)
  const hasMemories = (turn.memories?.length ?? 0) > 0
  const { colors } = useTheme()

  return (
    <div className="flex flex-col gap-1 max-w-[80%]">
      {/* bubble */}
      <div
        className="rounded px-4 py-3 text-sm leading-relaxed text-neutral-200 font-mono whitespace-pre-wrap"
        style={{
          background: `linear-gradient(135deg, ${colors.primaryDim} 0%, rgba(0,0,0,0.4) 100%)`,
          border: `1px solid ${colors.primaryBorder}`,
          boxShadow: `0 0 12px ${colors.primaryDim}`,
        }}
      >
        {turn.error ? (
          <span className="text-red-400">{turn.content}</span>
        ) : (
          turn.content
        )}
      </div>

      {/* meta row */}
      <div className="flex items-center gap-3 px-1">
        {turn.model && (
          <span className="text-[9px] text-neutral-700 font-mono">{turn.model}</span>
        )}
        {turn.tokens && (
          <span className="text-[9px] text-neutral-700 font-mono tabular-nums">
            {turn.tokens.input}→{turn.tokens.output} tok
          </span>
        )}
        {turn.retrieval_ms !== undefined && (
          <span className="text-[9px] text-neutral-700 font-mono tabular-nums">
            {turn.retrieval_ms}ms
          </span>
        )}
        {hasMemories && (
          <button
            onClick={() => setOpen(v => !v)}
            className="ml-auto flex items-center gap-1 text-[9px] hover:text-emerald-400 transition-colors font-mono uppercase tracking-wider"
            style={{ color: colors.primaryTextDim }}
          >
            <Brain className="w-2.5 h-2.5" />
            {turn.memories!.length} mem
            {open ? <ChevronUp className="w-2.5 h-2.5" /> : <ChevronDown className="w-2.5 h-2.5" />}
          </button>
        )}
      </div>

      {/* memory panel */}
      {open && hasMemories && (
        <div
          className="rounded px-3 py-2 space-y-1.5"
          style={{
            background: "rgba(0,0,0,0.25)",
            border: `1px solid ${colors.primaryBorder}`,
          }}
        >
          <div className="text-[8px] uppercase tracking-widest mb-2" style={{ color: colors.primaryTextDim }}>
            Retrieved memories
          </div>
          {turn.memories!.map((m, i) => (
            <MemoryBadge key={i} m={m} />
          ))}
        </div>
      )}
    </div>
  )
}

function UserBubble({ turn }: { turn: Turn }) {
  return (
    <div
      className="self-end max-w-[80%] rounded px-4 py-3 text-sm leading-relaxed font-mono whitespace-pre-wrap"
      style={{
        background: "linear-gradient(135deg, rgba(255,140,38,0.10) 0%, rgba(0,0,0,0.4) 100%)",
        border: "1px solid rgba(255,140,38,0.18)",
        color: "rgba(255,220,180,0.9)",
      }}
    >
      {turn.content}
    </div>
  )
}

export function ChatPage() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const { colors } = useTheme()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [turns, loading])

  const history: ChatMessage[] = turns
    .filter(t => !t.error)
    .map(t => ({ role: t.role, content: t.content }))

  async function send() {
    const msg = input.trim()
    if (!msg || loading) return

    setInput("")
    setTurns(prev => [...prev, { role: "user", content: msg }])
    setLoading(true)

    // Add placeholder for streaming response
    setTurns(prev => [
      ...prev,
      {
        role: "assistant",
        content: "",
        memories: [],
        model: undefined,
        tokens: undefined,
        retrieval_ms: undefined,
      },
    ])

    let streamedContent = ""
    let metadata = { retrieval_ms: 0, memories_used: 0 }

    try {
      for await (const event of api.chatStream(msg, history)) {
        if (event.type === "metadata") {
          metadata = { retrieval_ms: event.retrieval_ms, memories_used: event.memories_used }
        } else if (event.type === "token") {
          streamedContent += event.content
          // Update the last turn with streamed content
          setTurns(prev => {
            const newTurns = [...prev]
            const lastTurn = newTurns[newTurns.length - 1]
            if (lastTurn && lastTurn.role === "assistant") {
              lastTurn.content = streamedContent
            }
            return newTurns
          })
        } else if (event.type === "done") {
          break
        } else if (event.type === "error") {
          throw new Error(event.content)
        }
      }

      // Fetch full response for memories and metadata
      const resp: ChatResponse = await api.chat(msg, history)
      setTurns(prev => {
        const newTurns = [...prev]
        const lastTurn = newTurns[newTurns.length - 1]
        if (lastTurn && lastTurn.role === "assistant") {
          lastTurn.content = resp.response
          lastTurn.memories = resp.memories_used
          lastTurn.model = resp.model
          lastTurn.tokens = resp.tokens
          lastTurn.retrieval_ms = resp.retrieval_ms
        }
        return newTurns
      })
    } catch (e: any) {
      setTurns(prev => {
        const newTurns = [...prev]
        const lastTurn = newTurns[newTurns.length - 1]
        if (lastTurn && lastTurn.role === "assistant") {
          lastTurn.content = e?.message ?? "Request failed"
          lastTurn.error = true
        }
        return newTurns
      })
    } finally {
      setLoading(false)
      textareaRef.current?.focus()
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="flex flex-col h-full font-mono" style={{ background: "#020d0d" }}>
      {/* header */}
      <div
        className="shrink-0 px-6 py-3 flex items-center gap-3"
        style={{ borderBottom: `1px solid ${colors.primaryBorder}` }}
      >
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4" style={{ filter: `drop-shadow(0 0 4px ${colors.primaryGlow})`, color: colors.primary }} />
          <span className="text-sm tracking-widest uppercase" style={{ color: colors.primaryText }}>Chat</span>
        </div>
        <div className="h-px flex-1" style={{ background: `linear-gradient(to right, ${colors.primaryDim}, transparent)` }} />
        <span className="text-[9px] text-neutral-700 uppercase tracking-widest">Memory-augmented LLM</span>
      </div>

      {/* messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {turns.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4 opacity-40">
            <Brain className="w-10 h-10 text-emerald-700" />
            <p className="text-[11px] text-neutral-600 uppercase tracking-widest text-center">
              Start a conversation<br />memories will be retrieved automatically
            </p>
          </div>
        )}

        {turns.map((turn, i) =>
          turn.role === "user" ? (
            <div key={i} className="flex justify-end">
              <UserBubble turn={turn} />
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <AssistantBubble turn={turn} />
            </div>
          )
        )}

        {loading && (
          <div className="flex justify-start">
            <div
              className="flex items-center gap-2 px-4 py-3 rounded text-xs"
              style={{ border: `1px solid ${colors.primaryBorder}`, background: colors.primaryDim, color: colors.primaryText }}
            >
              <Loader2 className="w-3 h-3 animate-spin" />
              <span className="tracking-widest uppercase text-[9px]">Thinking…</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* input bar */}
      <div
        className="shrink-0 px-4 py-3"
        style={{ borderTop: `1px solid ${colors.primaryBorder}` }}
      >
        <div
          className="flex items-end gap-2 rounded px-3 py-2"
          style={{
            background: colors.primaryDim,
            border: `1px solid ${colors.primaryBorder}`,
          }}
        >
          <textarea
            ref={textareaRef}
            className={cn(
              "flex-1 resize-none bg-transparent text-sm text-neutral-200 placeholder-neutral-700",
              "outline-none leading-relaxed max-h-36 font-mono"
            )}
            rows={1}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            style={{ fieldSizing: "content" } as any}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className={cn(
              "shrink-0 p-2 rounded transition-all duration-200",
              loading || !input.trim()
                ? "text-neutral-700 cursor-not-allowed"
                : ""
            )}
            style={
              !loading && input.trim()
                ? { color: colors.primaryText, filter: `drop-shadow(0 0 4px ${colors.primaryGlow})` }
                : {}
            }
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="text-[8px] text-neutral-800 mt-1 px-1">
          Shift+Enter for newline · memories retrieved automatically · facts extracted in background
        </p>
      </div>
    </div>
  )
}
