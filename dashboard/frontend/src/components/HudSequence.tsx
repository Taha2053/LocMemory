import { useEffect, useState } from "react"

interface BootSequenceProps {
  children: React.ReactNode
  isVisible: boolean
}

export function BootSequence({ children, isVisible }: BootSequenceProps) {
  const [phase, setPhase] = useState<"hidden" | "glitch" | "visible">("hidden")

  useEffect(() => {
    if (!isVisible) return
    setPhase("hidden")
    const t1 = setTimeout(() => setPhase("glitch"), 100)
    const t2 = setTimeout(() => setPhase("visible"), 800)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [isVisible])

  if (phase === "hidden") return null

  return (
    <div className={`hud-boot-${phase}`}>
      {children}
    </div>
  )
}