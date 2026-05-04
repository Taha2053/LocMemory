import { useEffect, useState } from "react"
import { api, type Domain } from "@/lib/api"
import { HudPanel, StatusDot } from "@/components/hud"
import { domainColor } from "@/lib/domainColors"

export function DomainsPanel() {
  const [domains, setDomains] = useState<Domain[]>([])

  useEffect(() => {
    api.domains()
      .then((data) =>
        setDomains(
          data.filter((d) => d.total > 0).sort((a, b) => b.total - a.total),
        ),
      )
      .catch(() => {})
  }, [])

  const total = domains.reduce((s, d) => s + d.total, 0)
  const top = domains.slice(0, 6)
  const maxCount = top.length > 0 ? top[0].total : 1

  return (
    <HudPanel id="Knowledge Domains" className="hud-panel hud-panel-2 p-4" progressValue={top.length > 0 ? 72 : 10}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-[10px] uppercase tracking-widest text-neutral-400">
          Knowledge Domains
        </div>
        <StatusDot
          label={total > 0 ? "MAPPED" : "EMPTY"}
          color={total > 0 ? "#00ff88" : "#525252"}
        />
      </div>

      {top.length === 0 ? (
        <div className="text-[10px] text-neutral-600 italic">no domains yet</div>
      ) : (
        <div className="space-y-2.5">
          {top.map((d, idx) => {
            const color = domainColor(d.name, idx)
            const pct = (d.total / maxCount) * 100
            return (
              <div key={d.name}>
                <div className="flex items-center justify-between text-[10px] mb-1">
                  <span className="flex items-center gap-1.5 text-neutral-200 capitalize">
                    <span
                      className="inline-block h-2 w-2 rounded-full shrink-0"
                      style={{ background: color, boxShadow: `0 0 6px ${color}` }}
                    />
                    {d.name}
                  </span>
                  <span className="text-neutral-500 tabular-nums font-mono">{d.total}</span>
                </div>
                <div className="h-[3px] bg-white/5 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{
                      width: `${pct}%`,
                      background: color,
                      boxShadow: `0 0 6px ${color}80`,
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {total > 0 && (
        <div className="mt-3 pt-2 border-t border-white/5 flex items-center justify-between text-[10px]">
          <span className="text-neutral-500 uppercase tracking-wider">total</span>
          <span className="text-neutral-200 tabular-nums font-mono">{total} nodes</span>
        </div>
      )}
    </HudPanel>
  )
}
