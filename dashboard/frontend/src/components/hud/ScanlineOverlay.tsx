export function ScanlineOverlay() {
  return (
    <div className="pointer-events-none absolute inset-0 z-[3] overflow-hidden opacity-30">
      <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_79px,rgba(0, 255, 136,0.03)_79px,rgba(0, 255, 136,0.03)_80px)]" />
      <div className="scanline-sweep absolute left-0 h-[2px] w-24 bg-gradient-to-r from-emerald-400/60 via-emerald-400/20 to-transparent opacity-60" />
    </div>
  )
}