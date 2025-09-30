# NOTE: Env must be imported here in order to trigger metaclass registering
from .envs.puzzle_agent.puzzle_tool import PuzzleToolBox
from .parallel_env import agent_rollout_loop
