

import { useState } from "react"
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Layout } from "@/components/layout"
import { GraphPage } from "@/pages/graph"
import { MemoriesPage } from "@/pages/memories"
import { DomainsPage } from "@/pages/domains"
import { RetrievalPage } from "@/pages/retrieval"
import { SettingsPage } from "@/pages/settings"
import { MetricsPage } from "@/pages/metrics"
import { MatrixIntro } from "@/components/MatrixIntro"

export default function App() {
  const [introComplete, setIntroComplete] = useState(false)

  return (
    <>
      {!introComplete && <MatrixIntro onComplete={() => setIntroComplete(true)} />}
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
            </Route>
          </Routes>
        </BrowserRouter>
      )}
    </>
  )
}
