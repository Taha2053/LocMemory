import { cn } from "@/lib/utils"

export function Progress({
  value = 0,
  className,
  barClassName,
}: {
  value?: number
  className?: string
  barClassName?: string
}) {
  const pct = Math.max(0, Math.min(100, value))
  return (
    <div className={cn("relative h-2 w-full overflow-hidden rounded-full bg-secondary", className)}>
      <div
        className={cn("h-full transition-all bg-primary", barClassName)}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
