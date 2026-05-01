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
      className={`hud-panel rounded-xl ring-1 ring-emerald-500/10 ${className}`}
      style={{
        transform: "translateZ(0)",
        background: "rgba(0, 18, 9, 0.65)",
        backdropFilter: "blur(14px) saturate(160%)",
        border: "1px solid rgba(0, 255, 136, 0.14)",
        borderRadius: "14px",
        boxShadow:
          "inset 0 1px 0 rgba(0, 255, 136, 0.18), 0 0 28px rgba(0, 255, 136, 0.07), 0 8px 32px rgba(0, 0, 0, 0.45)",
      }}
    >
      {id && (
        <div
          className="mb-2 flex items-center justify-between pb-2"
          style={{ borderBottom: "1px solid rgba(0, 255, 136, 0.08)" }}
        >
          <span className="text-hud-label text-[10px] font-mono uppercase tracking-widest text-emerald-400/70">
            // {id}
          </span>
          <div className="flex items-center gap-2">
            <div
              className="h-1 w-8 overflow-hidden"
              style={{ borderRadius: "999px", background: "rgba(0, 255, 136, 0.12)" }}
            >
              <div
                className="h-full hud-progress-fill"
                style={{
                  width: `${progressValue ?? 100}%`,
                  background: "rgba(0, 255, 136, 0.55)",
                  borderRadius: "999px",
                }}
              />
            </div>
          </div>
        </div>
      )}
      {children}
    </div>
  )
}