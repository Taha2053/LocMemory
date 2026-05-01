interface StatusDotProps {
  label?: string
  color?: string
}

export function StatusDot({
  label = "ONLINE",
  color = "#22d3ee",
}: StatusDotProps) {
  return (
    <div className="flex items-center gap-1.5">
      <span
        className="h-1.5 w-1.5 rounded-full hud-pulse"
        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}
      />
      <span className="text-[9px] font-mono uppercase tracking-widest text-emerald-400/60">
        {label}
      </span>
    </div>
  )
}