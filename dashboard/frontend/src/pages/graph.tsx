import { useEffect, useMemo, useRef, useState } from "react"
import ForceGraph2D from "react-force-graph-2d"
import { api, type GraphNode, type GraphLink } from "@/lib/api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { X } from "lucide-react"

const DOMAIN_COLORS: Record<string, string> = {
  health: "#84cc16",        // olive-green (like the pic's root hubs)
  engineering: "#d946ef",   // magenta
  programming: "#ef4444",   // pink-red
  work: "#f59e0b",          // amber
  personal: "#ec4899",      // pink
  finance: "#10b981",       // emerald
  learning: "#eab308",      // yellow
  social: "#a3a3a3",        // gray
}

const DEFAULT_COLOR = "#f59e0b"

function colorFor(domain: string): string {
  return DOMAIN_COLORS[domain] || DEFAULT_COLOR
}

export function GraphPage() {
  const [data, setData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({
    nodes: [],
    links: [],
  })
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const fgRef = useRef<any>(null)
  const [dims, setDims] = useState({ w: 800, h: 600 })

  useEffect(() => {
    api.graph().then(setData)
  }, [])

  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        setDims({
          w: containerRef.current.clientWidth,
          h: containerRef.current.clientHeight,
        })
      }
    }
    measure()
    window.addEventListener("resize", measure)
    return () => window.removeEventListener("resize", measure)
  }, [])

  // Precompute node degree — this drives the hub-and-spoke sizing
  const { nodes, links, degreeMap } = useMemo(() => {
    const degreeMap = new Map<string, number>()
    for (const l of data.links) {
      const s = typeof l.source === "string" ? l.source : (l.source as any)?.id
      const t = typeof l.target === "string" ? l.target : (l.target as any)?.id
      if (s) degreeMap.set(s, (degreeMap.get(s) ?? 0) + 1)
      if (t) degreeMap.set(t, (degreeMap.get(t) ?? 0) + 1)
    }
    return { nodes: data.nodes, links: data.links, degreeMap }
  }, [data])

  // Tune d3 forces for the "fireworks" layout
  useEffect(() => {
    if (!fgRef.current) return
    const fg = fgRef.current
    fg.d3Force("link")?.distance(30).strength(0.4)
    fg.d3Force("charge")?.strength(-40)
  }, [nodes.length])

  // Legend entries — only domains that actually appear
  const activeDomains = useMemo(() => {
    const set = new Set(nodes.map((n) => n.domain).filter(Boolean))
    return Array.from(set).sort()
  }, [nodes])

  const radius = (nodeId: string) => {
    const d = degreeMap.get(nodeId) ?? 0
    return Math.min(14, 2.2 + Math.sqrt(d) * 1.6)
  }

  return (
    <div ref={containerRef} className="relative h-screen w-full bg-[#0a0a0a] overflow-hidden">
      {/* Subtle header */}
      <header className="pointer-events-none absolute top-0 left-0 z-10 p-5">
        <h1 className="text-base font-medium text-neutral-300">Graph</h1>
        <p className="text-xs text-neutral-500">
          {nodes.length} nodes · {links.length} edges
        </p>
      </header>

      <ForceGraph2D
        ref={fgRef}
        width={dims.w}
        height={dims.h}
        backgroundColor="#0a0a0a"
        graphData={{ nodes, links }}
        // Node rendering — flat circle, size by degree, label gated by zoom/hover
        nodeCanvasObjectMode={() => "replace"}
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          const r = radius(node.id)
          const isHovered = hovered === node.id
          const isSelected = selected?.id === node.id

          ctx.beginPath()
          ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false)
          ctx.fillStyle = colorFor(node.domain)
          ctx.fill()

          // Soft halo when hovered/selected
          if (isHovered || isSelected) {
            ctx.beginPath()
            ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI, false)
            ctx.strokeStyle = "rgba(255,255,255,0.6)"
            ctx.lineWidth = 1.2
            ctx.stroke()
          }

          // Labels: only when zoomed in close, or for this node on hover
          const showLabel = globalScale > 2.5 || isHovered || isSelected
          if (showLabel && node.text) {
            const label = String(node.text).slice(0, 40)
            const fontSize = Math.max(4, 10 / globalScale)
            ctx.font = `${fontSize}px sans-serif`
            ctx.textAlign = "center"
            ctx.textBaseline = "top"
            ctx.fillStyle = "rgba(229,229,229,0.9)"
            ctx.fillText(label, node.x, node.y + r + 2)
          }
        }}
        nodePointerAreaPaint={(node: any, color, ctx) => {
          ctx.fillStyle = color
          ctx.beginPath()
          ctx.arc(node.x, node.y, radius(node.id) + 3, 0, 2 * Math.PI, false)
          ctx.fill()
        }}
        // Edges — thin, pale, soft
        linkColor={() => "rgba(180,180,180,0.15)"}
        linkWidth={0.5}
        linkDirectionalParticles={0}
        onNodeHover={(n: any) => setHovered(n?.id ?? null)}
        onNodeClick={(n: any) => setSelected(n as GraphNode)}
        onBackgroundClick={() => setSelected(null)}
        cooldownTicks={200}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
      />

      {/* Legend — bottom left, subtle */}
      {activeDomains.length > 0 && (
        <div className="absolute bottom-5 left-5 z-10 flex flex-col gap-1.5 rounded-md bg-black/50 backdrop-blur-sm px-3 py-2 border border-white/5">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-0.5">domains</div>
          {activeDomains.map((d) => (
            <div key={d} className="flex items-center gap-2 text-xs text-neutral-300">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: colorFor(d) }}
              />
              {d}
            </div>
          ))}
        </div>
      )}

      {/* Detail panel */}
      {selected && (
        <div className="absolute top-5 right-5 z-20 w-80 rounded-lg border border-white/10 bg-neutral-900/95 backdrop-blur-md p-4 space-y-3 shadow-xl">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: colorFor(selected.domain) }}
              />
              <span className="text-xs uppercase tracking-wider text-neutral-400">
                {selected.tier_name}
              </span>
            </div>
            <Button size="icon" variant="ghost" onClick={() => setSelected(null)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-sm text-neutral-100 leading-relaxed">{selected.text}</p>
          <div className="flex flex-wrap gap-1.5">
            {selected.domain && <Badge variant="outline">{selected.domain}</Badge>}
            {selected.subdomain && <Badge variant="outline">{selected.subdomain}</Badge>}
          </div>
          <div className="text-[10px] text-neutral-500 font-mono">
            {selected.created_at}
          </div>
          <div className="text-[10px] text-neutral-500">
            {degreeMap.get(selected.id) ?? 0} connections
          </div>
        </div>
      )}
    </div>
  )
}
