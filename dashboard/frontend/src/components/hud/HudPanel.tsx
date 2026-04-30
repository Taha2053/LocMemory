import { ReactNode } from "react"

interface HudPanelProps {
  children: ReactNode
  className?: string
  id?: string
  progressValue?: number
}

export function HudPanel({
  children,
  className = "",
  id,
  progressValue,
}: HudPanelProps) {
  return (
    <div
      className={`hud-panel rounded-lg border border-cyan-400/20 bg-black/60 bg-glass-gradient backdrop-blur-md shadow-hud-glow ring-1 ring-cyan-500/10 ${className}`}
      style={{ transform: "translateZ(0)" }}
    >
      {id && (
        <div className="mb-2 flex items-center justify-between border-b border-cyan-400/10 pb-2">
          <span className="text-hud-label text-[10px] font-mono uppercase tracking-widest text-cyan-500/70">
            // {id}
          </span>
          <div className="flex items-center gap-2">
            <div className="h-1 w-8 overflow-hidden rounded-full bg-cyan-400/20">
              <div
                className="h-full bg-cyan-400/60 hud-progress-fill"
                style={{ width: `${progressValue ?? 100}%` }}
              />
            </div>
          </div>
        </div>
      )}
      {children}
    </div>
  )
}