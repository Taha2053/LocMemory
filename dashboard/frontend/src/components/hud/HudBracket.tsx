interface HudBracketProps {
  position: "tl" | "tr" | "bl" | "br"
  size?: number
}

export function HudBracket({
  position,
  size = 48,
}: HudBracketProps) {
  const posMap = {
    tl: "top-3 left-3 border-t-2 border-l-2 rounded-tl-md",
    tr: "top-3 right-3 border-t-2 border-r-2 rounded-tr-md",
    bl: "bottom-3 left-3 border-b-2 border-l-2 rounded-bl-md",
    br: "bottom-3 right-3 border-b-2 border-r-2 rounded-br-md",
  }

  return (
    <div
      className={`pointer-events-none absolute z-20 h-[${size}px] w-[${size}px] border-emerald-400/60 ${posMap[position]} hud-bracket-draw`}
      style={{
        height: size,
        width: size,
        filter: "drop-shadow(0 0 6px rgba(0, 255, 136,0.7))",
      }}
    />
  )
}