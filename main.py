from theoremforge.manager import TheoremForgeStateManager
import asyncio
import re


def erase_header(code: str) -> str:
    import_pattern = re.compile(r"^import\s+.*$", re.MULTILINE)
    open_pattern = re.compile(r"^open\s+.*$", re.MULTILINE)
    set_option_pattern = re.compile(r"^set_option\s+.*$", re.MULTILINE)
    return re.sub(import_pattern, "", re.sub(open_pattern, "", re.sub(set_option_pattern, "", code)))

statement = """
theorem putnam_1997_a4
(G : Type*)
[Group G]
(φ : G → G)
(hφ : ∀ g1 g2 g3 h1 h2 h3 : G, (g1 * g2 * g3 = 1 ∧ h1 * h2 * h3 = 1) → φ g1 * φ g2 * φ g3 = φ h1 * φ h2 * φ h3)
: ∃ a : G, let ψ := fun g => a * φ g; ∀ x y : G, ψ (x * y) = ψ x * ψ y :=
sorry
"""

async def main():
    manager = TheoremForgeStateManager(max_workers=1, output_file="state_trace.json")
    await manager.start()
    await manager.submit_formal_statement(statement)
    await manager.wait_for_completion()

if __name__ == "__main__":
    asyncio.run(main())
