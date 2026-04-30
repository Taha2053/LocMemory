interface DataFeedLineProps {
  startX: number
  startY: number
  endX: number
  endY: number
  delay?: number
}

export function DataFeedLine({
  startX,
  startY,
  endX,
  endY,
  delay = 0,
}: DataFeedLineProps) {
  const length = Math.sqrt(
    Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2)
  )
  const angle = Math.atan2(endY - startY, endX - startX) * (180 / Math.PI)

  return (
    <svg
      className="pointer-events-none absolute inset-0 z-[4] overflow-visible"
      style={{ width: "100%", height: "100%" }}
    >
      <line
        x1={startX}
        y1={startY}
        x2={endX}
        y2={endY}
        stroke="#22d3ee"
        strokeWidth="1"
        strokeDasharray={length}
        strokeDashoffset={length}
        opacity="0.25"
        className="data-feed-line"
        style={{
          transformOrigin: `${startX}px ${startY}px`,
          animationDelay: `${delay}ms`,
          transform: `rotate(${angle}deg)`,
        }}
      />
      <circle
        cx={endX}
        cy={endY}
        r="2"
        fill="#22d3ee"
        opacity="0"
        className="data-feed-dot"
        style={{ animationDelay: `${delay + 400}ms` }}
      />
    </svg>
  )
}