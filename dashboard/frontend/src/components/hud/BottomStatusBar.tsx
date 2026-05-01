export function BottomStatusBar() {
  return (
    <div className="pointer-events-none absolute bottom-0 left-0 right-0 z-20 flex h-7 items-center justify-between border-t border-emerald-400/20 bg-black/40 px-4 backdrop-blur-sm">
      <div className="flex items-center gap-4">
        <span className="text-[10px] font-mono text-emerald-400/50">
          FPS: <span className="text-emerald-400/80">60</span>
        </span>
        <span className="text-[10px] font-mono text-emerald-400/50">
          LAT: <span className="text-emerald-400/80">4ms</span>
        </span>
      </div>

      <Waveform />

      <div className="flex items-center gap-4">
        <span className="text-[10px] font-mono text-green-400/80">
          NEURAL LINK: STABLE
        </span>
        <span className="text-[10px] font-mono text-emerald-400/40">
          NODE: /memory/graph
        </span>
      </div>
    </div>
  )
}

function Waveform() {
  const bars = [40, 70, 50, 90, 60, 80, 45, 75, 55, 85]
  return (
    <div className="flex h-3 items-end gap-[2px]">
      {bars.map((h, i) => (
        <div
          key={i}
          className="w-[3px] bg-emerald-400/50 hud-wave-bar"
          style={{
            height: `${h}%`,
            animationDelay: `${i * 100}ms`,
          }}
        />
      ))}
    </div>
  )
}