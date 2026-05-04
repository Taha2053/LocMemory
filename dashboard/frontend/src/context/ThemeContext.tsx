import { createContext, useContext, useState, useEffect, ReactNode } from "react"

export type Theme = "teal" | "blue" | "purple"

interface ThemeColors {
  primary: string
  primaryDim: string
  primaryGlow: string
  primaryBorder: string
  primaryText: string
  primaryTextDim: string
}

const THEMES: Record<Theme, ThemeColors> = {
  teal: {
    primary: "#00c4bc",
    primaryDim: "rgba(0, 196, 188, 0.3)",
    primaryGlow: "rgba(0, 196, 188, 0.6)",
    primaryBorder: "rgba(0, 196, 188, 0.2)",
    primaryText: "#00c4bc",
    primaryTextDim: "rgba(0, 196, 188, 0.6)",
  },
  blue: {
    primary: "#3b82f6",
    primaryDim: "rgba(59, 130, 246, 0.3)",
    primaryGlow: "rgba(59, 130, 246, 0.6)",
    primaryBorder: "rgba(59, 130, 246, 0.2)",
    primaryText: "#3b82f6",
    primaryTextDim: "rgba(59, 130, 246, 0.6)",
  },
  purple: {
    primary: "#a855f7",
    primaryDim: "rgba(168, 85, 247, 0.3)",
    primaryGlow: "rgba(168, 85, 247, 0.6)",
    primaryBorder: "rgba(168, 85, 247, 0.2)",
    primaryText: "#a855f7",
    primaryTextDim: "rgba(168, 85, 247, 0.6)",
  },
}

interface ThemeContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  colors: ThemeColors
}

const ThemeContext = createContext<ThemeContextType | null>(null)

const STORAGE_KEY = "locmemory_theme"

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === "teal" || stored === "blue" || stored === "purple") {
      return stored
    }
    return "teal"
  })

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme)
    localStorage.setItem(STORAGE_KEY, newTheme)
  }

  const colors = THEMES[theme]

  useEffect(() => {
    // Set CSS custom properties for app-wide use
    const root = document.documentElement
    root.style.setProperty("--theme-primary", colors.primary)
    root.style.setProperty("--theme-primary-dim", colors.primaryDim)
    root.style.setProperty("--theme-primary-glow", colors.primaryGlow)
    root.style.setProperty("--theme-primary-border", colors.primaryBorder)
    root.style.setProperty("--theme-primary-text", colors.primaryText)
    root.style.setProperty("--theme-primary-text-dim", colors.primaryTextDim)
  }, [colors])

  return (
    <ThemeContext.Provider value={{ theme, setTheme, colors }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error("useTheme must be used within ThemeProvider")
  }
  return context
}