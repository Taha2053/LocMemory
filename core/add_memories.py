"""Add sample memories to get started"""

from core.memory import MemoryStore

store = MemoryStore(db_path="data/memories.db", md_dir="memories")

memories = [
    ("My name is Taha and I'm a software developer.", "fact"),
    ("I live in Paris, France.", "fact"),
    ("I love working with Python and AI.", "fact"),
    ("My favorite programming language is Python.", "fact"),
    ("I'm building a project called LocMemory for LLM memory.", "project"),
    ("I work as a full-stack developer.", "work"),
    ("I have a dog named Max.", "fact"),
    ("My birthday is on March 15th.", "fact"),
    ("I studied computer science at university.", "fact"),
    ("I'm learning about RAG and embeddings.", "learning"),
]

print("Adding memories...\n")
for text, category in memories:
    mem = store.add(text, category=category)
    print(f"✓ {mem.id[:8]}... [{category}]")

print(f"\n Added {len(memories)} memories!")
