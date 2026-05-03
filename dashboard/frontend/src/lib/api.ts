const BASE = "/api"

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
  return r.json() as Promise<T>
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
  density: number
  avg_degree: number
  communities: number
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
  entry_id: string | null
  results: RetrievedResult[]
  rejected?: RetrievedResult[]
}

export interface MetricsSummary {
  total_retrievals: number
  avg_result_count: number
  avg_score: number
  avg_keyword_overlap: number
  avg_latency_ms: number
  precision_at_5: number
  rated_count: number
  domain_distribution: Record<string, number>
  recent: {
    id: string
    timestamp: string
    query: string
    query_domain: string
    result_count: number
    avg_score: number
    keyword_overlap: number
    latency_ms: number
    user_rating: number | null
  }[]
}

export const api = {
  stats: (): Promise<Stats> => fetch(`${BASE}/stats`).then((r) => j<Stats>(r)),
  graph: (): Promise<{ nodes: GraphNode[]; links: GraphLink[] }> =>
    fetch(`${BASE}/graph`).then((r) => j(r)),
  memories: (params?: { domain?: string; subdomain?: string; tier?: number; q?: string; limit?: number; offset?: number }): Promise<Memory[]> => {
    const qs = new URLSearchParams()
    if (params?.domain) qs.set("domain", params.domain)
    if (params?.subdomain) qs.set("subdomain", params.subdomain)
    if (params?.tier !== undefined) qs.set("tier", String(params.tier))
    if (params?.q) qs.set("q", params.q)
    if (params?.limit) qs.set("limit", String(params.limit))
    if (params?.offset) qs.set("offset", String(params.offset))
    const suffix = qs.toString() ? `?${qs}` : ""
    return fetch(`${BASE}/memories${suffix}`).then((r) => j(r))
  },
  memory: (id: string): Promise<MemoryDetail> => fetch(`${BASE}/memories/${id}`).then((r) => j(r)),
  updateMemory: (id: string, text: string): Promise<Memory> =>
    fetch(`${BASE}/memories/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    }).then((r) => j(r)),
  deleteMemory: (id: string): Promise<{ deleted: string }> =>
    fetch(`${BASE}/memories/${id}`, { method: "DELETE" }).then((r) => j(r)),
  domains: (): Promise<Domain[]> => fetch(`${BASE}/domains`).then((r) => j(r)),
  retrieve: (query: string, limit = 10, includeRejected = false): Promise<RetrieveResponse> =>
    fetch(`${BASE}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit, include_rejected: includeRejected }),
    }).then((r) => j(r)),
  createMemory: (body: {
    text: string
    tier?: number
    domain?: string
    subdomain?: string
  }): Promise<Memory> =>
    fetch(`${BASE}/memories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => j(r)),
  hebbianStats: (): Promise<{
    count: number
    min_weight: number
    max_weight: number
    avg_weight: number
    histogram: { range: string; count: number }[]
  }> => fetch(`${BASE}/hebbian/stats`).then((r) => j(r)),
  hebbianDecay: (): Promise<{ edges_decayed: number }> =>
    fetch(`${BASE}/hebbian/decay`, { method: "POST" }).then((r) => j(r)),
  rlStatus: (): Promise<{
    enabled: boolean
    available: boolean
    model_path?: string
    candidate_pool_size?: number
    top_k?: number
    token_budget?: number
    message?: string
  }> => fetch(`${BASE}/rl/status`).then((r) => j(r)),
  consolidate: (): Promise<{
    clusters_found: number
    anchors_created: number
    nodes_connected: number
  }> => fetch(`${BASE}/consolidate`, { method: "POST" }).then((r) => j(r)),
  patterns: (): Promise<{ id: string; text: string; domain: string; created_at: string }[]> =>
    fetch(`${BASE}/patterns`).then((r) => j(r)),
  detectPatterns: (): Promise<{
    patterns_found: number
    procedural_nodes_created: number
  }> => fetch(`${BASE}/patterns/detect`, { method: "POST" }).then((r) => j(r)),
  config: (): Promise<Record<string, any>> => fetch(`${BASE}/config`).then((r) => j(r)),
  updateConfig: (data: Record<string, any>): Promise<{ ok: boolean; data: Record<string, any> }> =>
    fetch(`${BASE}/config`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data }),
    }).then((r) => j(r)),
  metrics: (n = 100): Promise<MetricsSummary> =>
    fetch(`${BASE}/metrics?n=${n}`).then((r) => j<MetricsSummary>(r)),
  rateRetrieval: (entryId: string, rating: number): Promise<{ ok: boolean; entry_id: string; rating: number }> =>
    fetch(`${BASE}/retrieve/${entryId}/rate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    }).then((r) => j(r)),
  rlTrain: (): Promise<{ ok: boolean; message: string }> =>
    fetch(`${BASE}/rl/train`, { method: "POST" }).then((r) => j(r)),
  rlTrainCancel: (): Promise<{ ok: boolean }> =>
    fetch(`${BASE}/rl/train/cancel`, { method: "POST" }).then((r) => j(r)),
  rlTrainStatus: (): Promise<{
    running: boolean
    progress: number
    total: number
    last_reward: number | null
    log: string[]
    done: boolean
    error: string | null
  }> => fetch(`${BASE}/rl/train/status`).then((r) => j(r)),
  rlReload: (): Promise<{ ok: boolean; available: boolean; message: string }> =>
    fetch(`${BASE}/rl/reload`, { method: "POST" }).then((r) => j(r)),
  compareRetrieve: (query: string, limit = 5): Promise<{
    query: string
    query_domain: string
    rl_available: boolean
    hybrid: RetrievedResult[]
    rl: RetrievedResult[]
    overlap_count: number
  }> =>
    fetch(`${BASE}/retrieve/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit }),
    }).then((r) => j(r)),
}
