"""
Seed the LocMemory database with realistic test data.

Usage:
    uv run python scripts/seed_db.py

Adds memories across all tiers and domains with edges between
related nodes so the graph, Hebbian panel, and retrieval all have
something meaningful to display.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory.graph import GraphManager, TIER_CONTEXT, TIER_ANCHOR, TIER_LEAF, TIER_PROCEDURAL
from core.memory.classifier import MemoryClassifier
from core.settings.config import get_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    config = get_config()
    db_path = config.get("storage", "sqlite_db_path", "data/memory.db")
    if not Path(db_path).is_absolute():
        db_path = str(PROJECT_ROOT / db_path)

    print(f"[seed] connecting to: {db_path}")
    classifier = MemoryClassifier(use_fallback=False)

    with GraphManager(db_path=db_path) as gm:
        before = gm.graph.number_of_nodes()
        print(f"[seed] graph before: {before} nodes, {gm.graph.number_of_edges()} edges")

        def add(text, tier, domain, subdomain="general"):
            emb = None
            try:
                emb = classifier._embed([text])[0]
            except Exception:
                pass
            nid = gm.add_node(text=text, tier=tier, domain=domain, subdomain=subdomain, embedding=emb)
            print(f"  + [{domain}/{subdomain}] T{tier}  {text[:60]}")
            return nid

        def link(a, b, weight=0.4, relation="related"):
            gm.add_edge(a, b, weight=weight, relation=relation)

        # ── Tier 1: Core Context (semantic hubs) ──────────────────────────
        c_health  = add("User maintains an active healthy lifestyle focused on fitness and nutrition",
                        TIER_CONTEXT, "health", "fitness")
        c_learn   = add("User is an active learner pursuing software engineering and AI topics",
                        TIER_CONTEXT, "learning", "technical")
        c_work    = add("User works as a software developer on backend systems",
                        TIER_CONTEXT, "work", "engineering")
        c_personal = add("User values personal growth, mindfulness, and work-life balance",
                         TIER_CONTEXT, "personal", "wellbeing")

        # ── Tier 2: Anchors (stable reference points) ─────────────────────
        a_running   = add("User runs 5km three times per week as main cardio exercise",
                          TIER_ANCHOR, "health", "fitness")
        a_diet      = add("User follows a high-protein low-carb diet for muscle recovery",
                          TIER_ANCHOR, "health", "nutrition")
        a_python    = add("User has 3 years of Python experience and uses FastAPI and SQLAlchemy",
                          TIER_ANCHOR, "learning", "programming")
        a_ml        = add("User is studying machine learning with focus on transformers and embeddings",
                          TIER_ANCHOR, "learning", "ai")
        a_job       = add("User works at a tech startup building a cognitive memory system",
                          TIER_ANCHOR, "work", "engineering")
        a_sleep     = add("User sleeps 7-8 hours per night and tracks sleep quality",
                          TIER_ANCHOR, "health", "sleep")
        a_meditation = add("User meditates 10 minutes every morning using Headspace",
                           TIER_ANCHOR, "personal", "mindfulness")

        # ── Tier 3: Leaves (atomic facts) ─────────────────────────────────
        l_run1   = add("User completed a 10km race last Sunday in 52 minutes",
                       TIER_LEAF, "health", "fitness")
        l_run2   = add("User prefers running in the morning before breakfast",
                       TIER_LEAF, "health", "fitness")
        l_run3   = add("User uses Nike running shoes and tracks workouts with Garmin",
                       TIER_LEAF, "health", "fitness")
        l_diet1  = add("User drinks 2 liters of water daily and avoids sugary drinks",
                       TIER_LEAF, "health", "nutrition")
        l_diet2  = add("User eats oatmeal with whey protein for breakfast every morning",
                       TIER_LEAF, "health", "nutrition")
        l_gym    = add("User lifts weights at the gym on Tuesday and Thursday",
                       TIER_LEAF, "health", "fitness")
        l_py1    = add("User built a REST API with FastAPI and deployed it on a VPS",
                       TIER_LEAF, "learning", "programming")
        l_py2    = add("User is reading the book 'Fluent Python' to deepen Python knowledge",
                       TIER_LEAF, "learning", "programming")
        l_ml1    = add("User completed the deeplearning.ai specialization on Coursera",
                       TIER_LEAF, "learning", "ai")
        l_ml2    = add("User is implementing a RAG pipeline using sentence-transformers and FAISS",
                       TIER_LEAF, "learning", "ai")
        l_ml3    = add("User fine-tuned a BERT model for text classification on a custom dataset",
                       TIER_LEAF, "learning", "ai")
        l_work1  = add("User had a code review session focusing on database query optimization",
                       TIER_LEAF, "work", "engineering")
        l_work2  = add("User is implementing a graph-based memory system with NetworkX and SQLite",
                       TIER_LEAF, "work", "engineering")
        l_work3  = add("User presented the architecture of LocMemory to the team last Thursday",
                       TIER_LEAF, "work", "engineering")
        l_med1   = add("User finds that meditation reduces stress before important meetings",
                       TIER_LEAF, "personal", "mindfulness")
        l_fin1   = add("User tracks monthly expenses using a spreadsheet",
                       TIER_LEAF, "finance", "budgeting")
        l_fin2   = add("User invests 10% of monthly income into an index fund",
                       TIER_LEAF, "finance", "investing")
        l_eng1   = add("User set up a CI/CD pipeline with GitHub Actions for the project",
                       TIER_LEAF, "engineering", "devops")
        l_eng2   = add("User uses Docker to containerize the backend and frontend services",
                       TIER_LEAF, "engineering", "devops")

        # ── Tier 4: Procedural (skills/workflows) ─────────────────────────
        p_deploy = add("Workflow: write code -> run tests -> build Docker image -> push to registry -> deploy via SSH",
                       TIER_PROCEDURAL, "engineering", "devops")
        p_study  = add("Study routine: read docs -> build small project -> write tests -> review -> repeat",
                       TIER_PROCEDURAL, "learning", "technical")
        p_fitness = add("Weekly fitness pattern: run Mon/Wed/Fri, lift Tue/Thu, rest Sat/Sun",
                        TIER_PROCEDURAL, "health", "fitness")

        # ── Edges (semantic relationships) ────────────────────────────────
        print("\n[seed] creating edges...")

        # Health cluster
        link(c_health,  a_running,   weight=0.85, relation="encompasses")
        link(c_health,  a_diet,      weight=0.80, relation="encompasses")
        link(c_health,  a_sleep,     weight=0.70, relation="encompasses")
        link(a_running, l_run1,      weight=0.90, relation="instance_of")
        link(a_running, l_run2,      weight=0.75, relation="instance_of")
        link(a_running, l_run3,      weight=0.60, relation="instance_of")
        link(a_running, l_gym,       weight=0.65, relation="related")
        link(a_diet,    l_diet1,     weight=0.80, relation="instance_of")
        link(a_diet,    l_diet2,     weight=0.85, relation="instance_of")
        link(l_run1,    l_run2,      weight=0.70, relation="related")
        link(l_diet1,   l_diet2,     weight=0.60, relation="related")
        link(a_running, p_fitness,   weight=0.80, relation="follows")

        # Learning cluster
        link(c_learn,   a_python,    weight=0.90, relation="encompasses")
        link(c_learn,   a_ml,        weight=0.88, relation="encompasses")
        link(a_python,  l_py1,       weight=0.85, relation="instance_of")
        link(a_python,  l_py2,       weight=0.75, relation="instance_of")
        link(a_ml,      l_ml1,       weight=0.80, relation="instance_of")
        link(a_ml,      l_ml2,       weight=0.90, relation="instance_of")
        link(a_ml,      l_ml3,       weight=0.85, relation="instance_of")
        link(l_py1,     l_work2,     weight=0.70, relation="applied_in")
        link(l_ml2,     l_work2,     weight=0.75, relation="applied_in")
        link(a_ml,      p_study,     weight=0.70, relation="follows")

        # Work cluster
        link(c_work,    a_job,       weight=0.95, relation="encompasses")
        link(a_job,     l_work1,     weight=0.80, relation="instance_of")
        link(a_job,     l_work2,     weight=0.90, relation="instance_of")
        link(a_job,     l_work3,     weight=0.75, relation="instance_of")
        link(l_work2,   l_eng1,      weight=0.65, relation="related")
        link(l_work2,   l_eng2,      weight=0.65, relation="related")
        link(l_eng1,    l_eng2,      weight=0.80, relation="related")
        link(l_eng1,    p_deploy,    weight=0.85, relation="follows")
        link(l_eng2,    p_deploy,    weight=0.80, relation="follows")

        # Personal cluster
        link(c_personal, a_meditation, weight=0.85, relation="encompasses")
        link(a_meditation, l_med1,    weight=0.90, relation="instance_of")
        link(a_sleep,    a_meditation, weight=0.55, relation="supports")

        # Cross-domain bridges (important for Hebbian + procedural detection)
        link(c_work,    c_learn,     weight=0.60, relation="reinforces")
        link(a_python,  a_job,       weight=0.80, relation="used_in")
        link(a_ml,      a_job,       weight=0.75, relation="used_in")
        link(c_health,  c_personal,  weight=0.65, relation="supports")
        link(a_running, l_med1,      weight=0.45, relation="related")
        link(l_fin1,    l_fin2,      weight=0.70, relation="related")
        link(a_sleep,   a_running,   weight=0.55, relation="enables")

        gm.save_graph()
        after_nodes = gm.graph.number_of_nodes()
        after_edges = gm.graph.number_of_edges()
        added = after_nodes - before

        print(f"\n[seed] done.")
        print(f"  nodes added : {added}")
        print(f"  total nodes : {after_nodes}")
        print(f"  total edges : {after_edges}")
        print(f"  tiers       : {TIER_CONTEXT}×context  {TIER_ANCHOR}×anchor  {TIER_LEAF}×leaf  {TIER_PROCEDURAL}×procedural")
        print(f"\nRestart the backend to reload the graph, then refresh the dashboard.")


if __name__ == "__main__":
    main()
