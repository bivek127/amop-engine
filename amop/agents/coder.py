from amop.agents.base import BaseAgent


class CoderAgent(BaseAgent):
    name = "coder"

    def system_prompt(self) -> str:
        return (
            "You are a code assistant. Given a description of a change, "
            "produce the code diff or implementation."
        )
