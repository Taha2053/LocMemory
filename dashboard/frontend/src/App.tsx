

import { useState } from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Layout } from "@/components/layout"
import { GraphPage } from "@/pages/graph"
import { MemoriesPage } from "@/pages/memories"
import { DomainsPage } from "@/pages/domains"
import { RetrievalPage } from "@/pages/retrieval"
import { SettingsPage } from "@/pages/settings"
import { MetricsPage } from "@/pages/metrics"
import { GuidePage } from "@/pages/guide"
import { ChatPage } from "@/pages/chat"
import { AboutPage } from "@/pages/about"
import { MatrixIntro } from "@/components/MatrixIntro"
import { ThemeProvider } from "@/context/ThemeContext"

export default function App() {
  const [introComplete, setIntroComplete] = useState(
    () => localStorage.getItem("locmemory_intro_seen") === "1"
  )

  const handleIntroComplete = () => {
    localStorage.setItem("locmemory_intro_seen", "1")
    setIntroComplete(true)
  }

  return (
    <ThemeProvider>
      <style>{`
        :root {
          --theme-primary: #00c4bc;
          --theme-primary-dim: rgba(0, 196, 188, 0.3);
          --theme-primary-glow: rgba(0, 196, 188, 0.6);
          --theme-primary-border: rgba(0, 196, 188, 0.2);
          --theme-primary-text: #00c4bc;
          --theme-primary-text-dim: rgba(0, 196, 188, 0.6);
        }
      `}</style>
      {!introComplete && <MatrixIntro onComplete={handleIntroComplete} />}
      {introComplete && (
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Navigate to="/memories" replace />} />
              <Route path="/graph" element={<GraphPage />} />
              <Route path="/memories" element={<MemoriesPage />} />
              <Route path="/domains" element={<DomainsPage />} />
              <Route path="/retrieval" element={<RetrievalPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/metrics" element={<MetricsPage />} />
              <Route path="/guide" element={<GuidePage />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      )}
    </ThemeProvider>
  )
}
