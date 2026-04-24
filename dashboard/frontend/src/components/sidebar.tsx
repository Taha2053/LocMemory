import { NavLink } from "react-router-dom"
import { Network, FileText, FolderTree, Search, Settings, Sun, Moon } from "lucide-react"
import { useEffect, useState } from "react"
import { api, type Stats } from "@/lib/api"
import { useTheme } from "@/lib/theme"
import { cn } from "@/lib/utils"

const links = [
  { to: "/graph", label: "Graph", icon: Network },
  { to: "/memories", label: "Memories", icon: FileText },
  { to: "/domains", label: "Domains", icon: FolderTree },
  { to: "/retrieval", label: "Retrieval", icon: Search },
  { to: "/settings", label: "Settings", icon: Settings },
]

export function Sidebar() {
  const [stats, setStats] = useState<Stats | null>(null)
  const { theme, toggle } = useTheme()

  useEffect(() => {
    api.stats().then(setStats).catch(() => {})
  }, [])

  return (
    <aside className="flex h-screen w-56 flex-col border-r border-border bg-card">
      <div className="border-b border-border px-5 py-4">
        <div className="text-sm font-semibold tracking-tight">LocMemory</div>
        <div className="text-xs text-muted-foreground">cognitive memory</div>
      </div>

      <nav className="flex-1 px-2 py-3 space-y-1">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-border px-3 py-3 space-y-2">
        <button
          onClick={toggle}
          className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          {theme === "dark" ? "Light mode" : "Dark mode"}
        </button>
        <div className="px-3 text-xs text-muted-foreground space-y-0.5">
          {stats ? (
            <>
              <div>{stats.nodes} nodes</div>
              <div>{stats.edges} edges</div>
            </>
          ) : (
            <div>loading…</div>
          )}
        </div>
      </div>
    </aside>
  )
}
