import { useEffect, useState } from "react"
import { api, type Domain } from "@/lib/api"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ChevronRight, ChevronDown } from "lucide-react"

export function DomainsPage() {
  const [domains, setDomains] = useState<Domain[]>([])
  const [open, setOpen] = useState<Record<string, boolean>>({})

  useEffect(() => {
    api.domains().then(setDomains)
  }, [])

  const toggle = (name: string) => setOpen((s) => ({ ...s, [name]: !s[name] }))

  return (
    <div className="p-6 space-y-4 max-w-3xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Domains</h1>
        <p className="text-sm text-muted-foreground">
          LLM-classified categories. Counts reflect memories stored per domain / subdomain.
        </p>
      </header>

      <div className="space-y-2">
        {domains.map((d) => {
          const isOpen = open[d.name] ?? false
          return (
            <Card key={d.name}>
              <CardContent className="p-0">
                <button
                  className="w-full p-4 flex items-center justify-between hover:bg-accent/50 transition-colors"
                  onClick={() => toggle(d.name)}
                >
                  <div className="flex items-center gap-2">
                    {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    <span className="font-medium">{d.name}</span>
                  </div>
                  <Badge variant="secondary">{d.total}</Badge>
                </button>
                {isOpen && (
                  <div className="border-t border-border px-4 py-2 space-y-1">
                    {d.subdomains.length === 0 ? (
                      <div className="text-xs text-muted-foreground py-2">no subdomains</div>
                    ) : (
                      d.subdomains.map((s) => (
                        <div key={s.name} className="flex items-center justify-between py-1 text-sm">
                          <span className="text-muted-foreground ml-6">{s.name}</span>
                          <Badge variant="outline">{s.count}</Badge>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
