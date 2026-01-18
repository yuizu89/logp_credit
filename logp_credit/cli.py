# logp_credit/cli.py
from __future__ import annotations

from logp_credit.config import ExperimentConfig, ModelConfig, GenConfig, DataConfig, RunConfig, ContributionConfig
from logp_credit.pipeline.runner import run_experiment


def main():
    # まずは最小で回るデフォルト設定
    cfg = ExperimentConfig(
        model=ModelConfig(
            model_name="Qwen/Qwen3-0.6B",
            attn_implementation="flash_attention_2",
            device_map=None,
        ),
        gen=GenConfig(max_new_tokens=512, temperature=0.6, top_p=0.95, top_k=20),
        data=DataConfig(split="test", idxs=[0, 1]),  # まずは2問
        run=RunConfig(seed=1, n_rollouts_per_question=2, save_level="light"),
        contrib=ContributionConfig(method="prefix"),
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
