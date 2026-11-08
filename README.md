GRPO vs SFT on tool use, reproducing the LLM part of
[RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/abs/2509.04259).
Training and evaluation go through [`orch`](https://github.com/thomas-woehrle/orch).

```bash
uv sync
uv run orch run experiments/tooluse_lr_sweep.yaml --dry-run
```
