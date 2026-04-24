import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/input"

export function SettingsPage() {
  const [cfg, setCfg] = useState<Record<string, any> | null>(null)
  const [text, setText] = useState("")
  const [status, setStatus] = useState<string>("")

  useEffect(() => {
    api.config().then((c) => {
      setCfg(c)
      setText(JSON.stringify(c, null, 2))
    })
  }, [])

  const save = async () => {
    setStatus("saving…")
    try {
      const parsed = JSON.parse(text)
      const res = await api.updateConfig(parsed)
      setCfg(res.data)
      setText(JSON.stringify(res.data, null, 2))
      setStatus("saved ✓")
    } catch (e: any) {
      setStatus(`error: ${e.message || e}`)
    }
  }

  const reset = () => {
    if (cfg) setText(JSON.stringify(cfg, null, 2))
    setStatus("")
  }

  return (
    <div className="p-6 space-y-4 max-w-4xl mx-auto">
      <header>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Edit <code className="text-xs bg-muted px-1 rounded">config.yaml</code> as JSON. Changes are written via <code className="text-xs bg-muted px-1 rounded">Config.save()</code>.
        </p>
      </header>

      {cfg === null ? (
        <div className="text-sm text-muted-foreground">loading…</div>
      ) : (
        <Card>
          <CardContent className="p-4 space-y-3">
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="min-h-[500px] font-mono text-xs"
              spellCheck={false}
            />
            <div className="flex items-center gap-2">
              <Button onClick={save}>Save</Button>
              <Button variant="outline" onClick={reset}>Reset</Button>
              <span className="text-sm text-muted-foreground">{status}</span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
