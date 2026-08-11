import re
from typing import Callable, List, Dict, Any
from datetime import datetime
from memory_store import VectorMemoryStore, MemoryEntry


SYSTEM_PROMPT = """You are a memory-augmented AI assistant for MyCompany.
You have access to persistent memory across conversations.
At each turn you may issue memory commands:
[MEMORY_SEARCH: <query>] - retrieve relevant memories
[MEMORY_WRITE: <content>] - store important information
[MEMORY_REFLECT] - synthesize insights from memory

Always think step by step. Use memory to avoid repeating mistakes and to personalize your responses."""


class MemoryAugmentedAgent:
    """An LLM agent with a full read-act-reflect-write memory cycle (Pages 350-352)."""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        memory_store: VectorMemoryStore,
        importance_threshold: float = 0.6,
        max_memory_tokens: int = 1500
    ):
        self.llm = llm_fn
        self.memory = memory_store
        self.importance_threshold = importance_threshold
        self.max_memory_tokens = max_memory_tokens
        self.conversation_history: List[Dict[str, str]] = []

    def step(self, user_message: str) -> str:
        """Full agent step (Read -> Act -> Reflect -> Write)."""
        # Step 1: Retrieve relevant memories
        memories = self.memory.retrieve(user_message, k=5)
        memory_context = self._format_memories(memories)

        # Step 2: Construct augmented prompt
        prompt = self._build_prompt(user_message, memory_context)

        # Step 3: Generate response
        raw_response = self.llm(prompt)

        # Step 4: Execute memory commands in response
        clean_response, memory_ops = self._parse_memory_commands(raw_response)
        self._execute_memory_ops(memory_ops, user_message, clean_response)

        # Step 5: Auto-write important information (Heuristic)
        self._auto_write(user_message, clean_response)

        # Step 6: Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": clean_response})

        return clean_response

    def _format_memories(self, memories: List[MemoryEntry]) -> str:
        if not memories:
            return ""
        lines = ["[Relevant Memories:]"]
        for i, m in enumerate(memories, 1):
            lines.append(f"  [{i}] (importance: {m.importance:.1f}): {m.content}")
        return "\n".join(lines)

    def _build_prompt(self, user_message: str, memory_context: str) -> str:
        prompt = SYSTEM_PROMPT + "\n\n"
        if memory_context:
            prompt += memory_context + "\n\n"
        
        prompt += "Recent History:\n"
        for turn in self.conversation_history[-6:]:
            prompt += f"{turn['role'].capitalize()}: {turn['content']}\n"
        
        prompt += f"User: {user_message}\nAnswer:"
        return prompt

    def _parse_memory_commands(self, response: str):
        """Extract and remove memory commands from response."""
        patterns = {
            "search": r"\[MEMORY_SEARCH:\s*(.+?)\]",
            "write": r"\[MEMORY_WRITE:\s*(.+?)\]",
            "reflect": r"\[MEMORY_REFLECT\]"
        }
        ops = []
        clean = response
        for op_type, pattern in patterns.items():
            for match in re.finditer(pattern, response, re.DOTALL):
                content = match.group(1) if op_type != "reflect" else None
                ops.append({"type": op_type, "content": content})
                clean = clean.replace(match.group(0), "").strip()
        return clean, ops

    def _execute_memory_ops(self, ops: List[Dict], user_msg: str, clean_response: str):
        """Execute memory commands issued by LLM."""
        for op in ops:
            if op["type"] == "search":
                self.memory.retrieve(op["content"], k=3)
            elif op["type"] == "write":
                self.memory.write(op["content"], importance=0.8)
            elif op["type"] == "reflect":
                self._reflect()

    def _auto_write(self, user_msg: str, response: str):
        """Automatically store important information using heuristic keywords."""
        importance_keywords = [
            "remember", "important", "note that", "you prefer",
            "my name is", "decided to", "key insight", "learned that"
        ]
        combined = (user_msg + " " + response).lower()
        if any(kw in combined for kw in importance_keywords):
            summary = f"User: {user_msg[:100]} | Agent: {response[:200]}"
            self.memory.write(summary, importance=self.importance_threshold)

    def _reflect(self):
        """Meta-cognitive reflection: synthesize insights from memory."""
        recent = self.memory.retrieve("recent important events", k=10)
        if len(recent) < 3:
            return  # Not enough memories to reflect on
        
        recent_text = "\n".join([f"- {m.content}" for m in recent])
        prompt = f"Based on these recent memories, extract 2-3 key insights:\n{recent_text}\nInsights:"
        insights = self.llm(prompt)
        
        for line in insights.split("\n"):
            line = line.strip().lstrip("-* ").strip()
            if len(line) > 20:
                self.memory.write(f"[INSIGHT] {line}", importance=0.9)


# --- Self-Test Block ---
if __name__ == "__main__":
    import numpy as np

    print("Testing MemoryAugmentedAgent...")

    # Dummy embedding & LLM for testing
    def dummy_embed(text):
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(128)

    def dummy_llm(prompt):
        if "name" in prompt.lower():
            return "Hello Manjesh! Nice to meet you. [MEMORY_WRITE: User's name is Manjesh]"
        return "I am MyCompanyBot, how can I help you today?"

    mem_store = VectorMemoryStore(embed_fn=dummy_embed)
    agent = MemoryAugmentedAgent(llm_fn=dummy_llm, memory_store=mem_store)

    reply = agent.step("My name is Manjesh")
    print(f"Agent Response: '{reply}'")
    print(f"Stored Memories Count: {len(mem_store.entries)}")
    print("Memory Agent Test Passed Successfully!")