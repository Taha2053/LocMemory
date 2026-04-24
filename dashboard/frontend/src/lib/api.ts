const BASE = "/api"

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json()
}

export interface Memory {
  id: string
  text: string
  tier: number
  tier_name: string
  domain: string
  subdomain: string
  created_at: string
}

export interface MemoryDetail extends Memory {
  neighbors: { id: string; text: string; relation?: string; weight?: number }[]
}

export interface GraphNode {
  id: string
  text: string
  tier: number
  tier_name: string
  domain: string
  subdomain: string
  created_at: string
}

export interface GraphLink {
  source: string
  target: string
  relation: string
  weight: number
}

export interface Stats {
  nodes: number
  edges: number
  tier_counts: Record<string, number>
  domain_counts: Record<string, number>
}

export interface Domain {
  name: string
  total: number
  subdomains: { name: string; count: number }[]
}

export interface RetrievedResult {
  node_id: string
  text: string
  domain: string
  subdomain: string
  tier: number
  score: number
  cosine: number
  recency: number
  category: number
  cosine_contribution: number
  recency_contribution: number
  category_contribution: number
  created_at: string
  depth: number
}

export interface RetrieveResponse {
  query: string
  query_domain: string
  results: RetrievedResult[]
}

export const api = {
  stats: (): Promise<Stats> => fetch(`${BASE}/stats`).then(j),
  graph: (): Promise<{ nodes: GraphNode[]; links: GraphLink[] }> =>
    fetch(`${BASE}/graph`).then(j),
  memories: (params?: { domain?: string; subdomain?: string; tier?: number; q?: string }): Promise<Memory[]> => {
    const qs = new URLSearchParams()
    if (params?.domain) qs.set("domain", params.domain)
    if (params?.subdomain) qs.set("subdomain", params.subdomain)
    if (params?.tier !== undefined) qs.set("tier", String(params.tier))
    if (params?.q) qs.set("q", params.q)
    const suffix = qs.toString() ? `?${qs}` : ""
    return fetch(`${BASE}/memories${suffix}`).then(j)
  },
  memory: (id: string): Promise<MemoryDetail> => fetch(`${BASE}/memories/${id}`).then(j),
  updateMemory: (id: string, text: string): Promise<Memory> =>
    fetch(`${BASE}/memories/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then(j),
  deleteMemory: (id: string): Promise<{ deleted: string }> =>
    fetch(`${BASE}/memories/${id}`, { method: "DELETE" }).then(j),
  domains: (): Promise<Domain[]> => fetch(`${BASE}/domains`).then(j),
  retrieve: (query: string, limit = 10): Promise<RetrieveResponse> =>
    fetch(`${BASE}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit }),
    }).then(j),
  config: (): Promise<Record<string, any>> => fetch(`${BASE}/config`).then(j),
  updateConfig: (data: Record<string, any>): Promise<{ ok: boolean; data: Record<string, any> }> =>
    fetch(`${BASE}/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data }),
    }).then(j),
}
