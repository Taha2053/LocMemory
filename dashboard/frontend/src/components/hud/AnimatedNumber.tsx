import { useEffect, useState } from "react"

interface AnimatedNumberProps {
  value: number
  duration?: number
  className?: string
}

export function AnimatedNumber({
  value,
  duration = 1200,
  className = "",
}: AnimatedNumberProps) {
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / duration, 1)
      const ease = 1 - Math.pow(1 - t, 3)
      setDisplay(Math.round(ease * value))
      if (t < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [value, duration])

  return (
    <span className={`tabular-nums ${className}`}>
      {display.toLocaleString()}
    </span>
  )
}