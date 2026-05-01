import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { ScanlineOverlay } from "@/components/hud"
import { Save, RotateCcw, CheckCircle, AlertCircle } from "lucide-react"

export function SettingsPage() {
  const [cfg, setCfg] = useState<Record<string, any> | null>(null)
  const [text, setText] = useState("")
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle")
  const [errorMsg, setErrorMsg] = useState("")
  const [isDirty, setIsDirty] = useState(false)

  useEffect(() => {
    api.config().then((c) => {
      setCfg(c)
      setText(JSON.stringify(c, null, 2))
    })
  }, [])

  const save = async () => {
    setStatus("saving")
    setErrorMsg("")
    try {
      const parsed = JSON.parse(text)
      const res = await api.updateConfig(parsed)
      setCfg(res.data)
      setText(JSON.stringify(res.data, null, 2))
      setStatus("saved")
      setIsDirty(false)
      setTimeout(() => setStatus("idle"), 2500)
    } catch (e: any) {
      setStatus("error")
      setErrorMsg(e.message || String(e))
    }
  }

  const reset = () => {
    if (cfg) {
      setText(JSON.stringify(cfg, null, 2))
      setIsDirty(false)
      setStatus("idle")
      setErrorMsg("")
    }
  }

  const handleChange = (val: string) => {
    setText(val)
    setIsDirty(val !== JSON.stringify(cfg, null, 2))
    if (status !== "idle") setStatus("idle")
  }

  return (
    <div className="relative h-full min-h-0 bg-[#020d0d] font-mono overflow-y-auto">
      <ScanlineOverlay />

      {/* Ambient glow */}
      <div className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 0% 0%, rgba(59, 200, 215,0.08), transparent 40%), radial-gradient(ellipse at 100% 100%, rgba(59, 160, 215,0.06), transparent 40%)" }} />

      {/* Corner brackets */}
      <div className="pointer-events-none absolute top-3 left-3 h-5 w-5 border-t-2 border-l-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 196, 188,0.5))" }} />
      <div className="pointer-events-none absolute top-3 right-3 h-5 w-5 border-t-2 border-r-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 196, 188,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 left-3 h-5 w-5 border-b-2 border-l-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 196, 188,0.5))" }} />
      <div className="pointer-events-none absolute bottom-3 right-3 h-5 w-5 border-b-2 border-r-2 border-emerald-400/40"
        style={{ filter: "drop-shadow(0 0 4px rgba(0, 196, 188,0.5))" }} />

      <div className="relative z-10 max-w-4xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            <div className="h-px w-8 bg-emerald-400/40" />
            <span className="text-[9px] uppercase tracking-[0.3em] text-emerald-600/60">// SYS.CONFIG</span>
          </div>
          <h1 className="text-2xl font-bold tracking-wide text-emerald-300"
            style={{ textShadow: "0 0 20px rgba(0, 196, 188,0.4)" }}>
            System Configuration
          </h1>
          <p className="mt-1 text-[11px] text-neutral-500 uppercase tracking-wider">
            Edit config.yaml as JSON — changes written via Config.save()
          </p>
        </div>

        {/* Editor panel */}
        {cfg === null ? (
          <div className="text-[10px] uppercase tracking-widest text-emerald-500/50 animate-pulse py-12 text-center">
            LOADING CONFIG...
          </div>
        ) : (
          <div className="border border-emerald-400/15 rounded-sm overflow-hidden"
            style={{
              background: "rgba(0,5,16,0.7)",
              boxShadow: "0 0 40px rgba(0, 196, 188,0.08), inset 0 0 30px rgba(0, 196, 188,0.03)"
            }}>

            {/* Panel header */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-emerald-400/10"
              style={{ background: "rgba(0, 196, 188,0.04)" }}>
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 rounded-full bg-emerald-400"
                  style={{ boxShadow: "0 0 6px rgba(0, 196, 188,0.8)", animation: "pulse 2s ease-in-out infinite" }} />
                <span className="text-[9px] uppercase tracking-[0.2em] text-emerald-500/70">
                  // CONFIG.EDITOR
                </span>
              </div>
              <div className="flex items-center gap-2">
                {isDirty && (
                  <span className="text-[8px] uppercase tracking-wider text-amber-400/70 px-2 py-0.5 border border-amber-400/20 rounded-sm">
                    UNSAVED
                  </span>
                )}
                {/* progress bar */}
                <div className="h-0.5 w-16 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-400/50 rounded-full" style={{ width: "100%" }} />
                </div>
              </div>
            </div>

            {/* Textarea */}
            <div className="relative">
              {/* Line numbers gutter */}
              <div className="absolute left-0 top-0 bottom-0 w-10 border-r border-emerald-400/8 pointer-events-none"
                style={{ background: "rgba(0, 196, 188,0.02)" }}>
                {text.split("\n").map((_, i) => (
                  <div key={i} className="h-[21px] flex items-center justify-center text-[9px] text-neutral-700 tabular-nums">
                    {i + 1}
                  </div>
                ))}
              </div>

              <textarea
                value={text}
                onChange={(e) => handleChange(e.target.value)}
                spellCheck={false}
                className="w-full min-h-[500px] bg-transparent text-[12px] text-neutral-200 focus:outline-none resize-none leading-[21px] pl-12 pr-4 py-4"
                style={{
                  caretColor: "#22d3ee",
                  fontFamily: "monospace",
                }}
              />
            </div>

            {/* Footer / actions */}
            <div className="flex items-center justify-between px-4 py-3 border-t border-emerald-400/10"
              style={{ background: "rgba(0, 196, 188,0.03)" }}>

              {/* Status message */}
              <div className="flex items-center gap-2 text-[10px]">
                {status === "saving" && (
                  <>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-emerald-400/70 uppercase tracking-wider">SAVING...</span>
                  </>
                )}
                {status === "saved" && (
                  <>
                    <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-green-400/80 uppercase tracking-wider">CONFIG SAVED</span>
                  </>
                )}
                {status === "error" && (
                  <>
                    <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                    <span className="text-red-400/80 uppercase tracking-wider truncate max-w-xs">{errorMsg}</span>
                  </>
                )}
              </div>

              {/* Buttons */}
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
          </div>
        )}
      </div>
    </div>
  )
}
