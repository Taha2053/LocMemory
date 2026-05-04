// Shared domain palette — used by BrainScene (3D nodes/edges),
// DomainsPanel (graph legend), and the Domains page.
// Vivid, well-separated hues so domains read clearly at a glance.

export const DOMAIN_COLORS: Record<string, string> = {
  health:      "#00ff7a",  // pure emerald
  programming: "#00d4ff",  // electric cyan
  work:        "#ff6a00",  // saturated orange
  learning:    "#b6ff00",  // hot lime
  personal:    "#b388ff",  // vivid lavender
  finance:     "#ffcc00",  // strong gold
  engineering: "#ff3355",  // crimson red
  social:      "#1e9eff",  // deep sky blue
}

export const FALLBACK_DOMAIN_COLORS = [
  "#00ff7a", "#00d4ff", "#ff6a00", "#b6ff00",
  "#b388ff", "#ffcc00", "#ff3355", "#1e9eff",
  "#9d33ff", "#ff44a0", "#22e9a3", "#ffae00",
]

// Stable hash of a domain string → fallback palette index.
function hashDomain(name: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i)
    h = (h * 0x01000193) >>> 0
  }
  return h
}

export function domainColor(name: string | undefined | null, idx?: number): string {
  if (!name) return FALLBACK_DOMAIN_COLORS[0]
  const exact = DOMAIN_COLORS[name.toLowerCase()]
  if (exact) return exact
  const i = idx !== undefined ? idx : hashDomain(name)
  return FALLBACK_DOMAIN_COLORS[i % FALLBACK_DOMAIN_COLORS.length]
}
