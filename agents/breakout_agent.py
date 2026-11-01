import asyncio
from agents.discoverer_agent import StrategyDiscovererAgent, generate_markdown
from config.settings import DATA_DIR
import sys

async def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    agent = StrategyDiscovererAgent("breakout")
    results = await agent.run(trials=trials)
    
    md_content = generate_markdown(results, "breakout")
    note_path = DATA_DIR / "Agent_Breakout_Brain.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
if __name__ == "__main__":
    asyncio.run(main())
