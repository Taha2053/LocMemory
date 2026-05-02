import { createContext, useContext, useState, type ReactNode } from "react"

interface PrivacyContextType {
  privacy: boolean
  toggle: () => void
}

const PrivacyContext = createContext<PrivacyContextType | null>(null)

export function PrivacyProvider({ children }: { children: ReactNode }) {
  const [privacy, setPrivacy] = useState(false)

  const toggle = () => setPrivacy((p) => !p)

  return (
    <PrivacyContext.Provider value={{ privacy, toggle }}>
      {children}
    </PrivacyContext.Provider>
  )
}

export function usePrivacy() {
  const ctx = useContext(PrivacyContext)
  if (!ctx) throw new Error("usePrivacy must be used within PrivacyProvider")
  return ctx
}