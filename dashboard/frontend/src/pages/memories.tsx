import { useEffect, useState } from "react"
import { api, type Memory } from "@/lib/api"
import { Card, CardContent } from "@/components/ui/card"
import { Input, Textarea } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Pencil, Check, X, Trash2 } from "lucide-react"

export function MemoriesPage() {
  const [items, setItems] = useState<Memory[]>([])
  const [loading, setLoading] = useState(true)
  const [q, setQ] = useState("")
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState("")

  const load = () => {
    setLoading(true)
    api.memories(q ? { q } : undefined)
      .then(setItems)
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [])

  const save = async (id: string) => {
    await api.updateMemory(id, draft)
    setEditingId(null)
    load()
  }

  const remove = async (id: string) => {
    if (!confirm("Delete this memory?")) return
    await api.deleteMemory(id)
    load()
  }

  return (
    <div className="p-6 space-y-4 max-w-5xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Memories</h1>
        <p className="text-sm text-muted-foreground">{items.length} entries · click the pencil to edit</p>
      </header>

      <div className="flex gap-2">
        <Input
          placeholder="Search memory text…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && load()}
        />
        <Button onClick={load} variant="outline">Search</Button>
      </div>

      {loading ? (
        <div className="text-sm text-muted-foreground">loading…</div>
      ) : (
        <div className="space-y-2">
          {items.map((m) => (
            <Card key={m.id}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="mb-1 flex flex-wrap items-center gap-1.5 text-xs">
                      <Badge variant="secondary">{m.tier_name}</Badge>
                      {m.domain && <Badge variant="outline">{m.domain}</Badge>}
                      {m.subdomain && <Badge variant="outline">{m.subdomain}</Badge>}
                      <span className="text-muted-foreground ml-1">{m.created_at.slice(0, 10)}</span>
                    </div>
                    {editingId === m.id ? (
                      <Textarea
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        autoFocus
                      />
                    ) : (
                      <p className="text-sm">{m.text}</p>
                    )}
                  </div>
                  <div className="flex gap-1 shrink-0">
                    {editingId === m.id ? (
                      <>
                        <Button size="icon" variant="ghost" onClick={() => save(m.id)}>
                          <Check className="h-4 w-4" />
                        </Button>
                        <Button size="icon" variant="ghost" onClick={() => setEditingId(null)}>
                          <X className="h-4 w-4" />
                        </Button>
                      </>
                    ) : (
                      <>
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => {
                            setEditingId(m.id)
                            setDraft(m.text)
                          }}
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                        <Button size="icon" variant="ghost" onClick={() => remove(m.id)}>
                          <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                      </>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
          {items.length === 0 && (
            <div className="text-sm text-muted-foreground">no memories match.</div>
          )}
        </div>
      )}
    </div>
  )
}
