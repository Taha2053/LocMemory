import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Layout } from "@/components/layout"
import { GraphPage } from "@/pages/graph"
import { MemoriesPage } from "@/pages/memories"
import { DomainsPage } from "@/pages/domains"
import { RetrievalPage } from "@/pages/retrieval"
import { SettingsPage } from "@/pages/settings"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/memories" replace />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/memories" element={<MemoriesPage />} />
          <Route path="/domains" element={<DomainsPage />} />
          <Route path="/retrieval" element={<RetrievalPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
