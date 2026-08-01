"""Single-GPU, suite-parallel Flow-Noise PPO for LIBERO."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm
from lerobot.envs import make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION

from .actor import FlowNoiseActor, executed_log_prob, trainable_parameters
from .critic import ValueCritic
from .ppo import gae, loss, normalize_advantage
from .rollout import (
    RolloutBuffer,
    SuiteSchedule,
    TaskResult,
    Transition,
    active_tasks,
    horizon,
    make_env,
    merge,
    suite_tasks,
    task_groups,
    worker_counts,
)


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def batch(observation, language, pre, env_pre, device):
    observation = preprocess_observation(observation)
    observation["task"] = list(language)
    return {
        key: value.to(device)
        for key, value in pre(env_pre(observation)).items()
        if isinstance(value, torch.Tensor)
    }


def amp(cfg):
    return torch.autocast("cuda", torch.bfloat16) if cfg.bfloat16 else nullcontext()


def log(message: str, file) -> None:
    """Print one training status line and persist the identical line."""
    print(message, flush=True)
    print(message, file=file, flush=True)


def compile_actor(actor):
    """Compile SmolVLA's prefix forward pass and action denoiser."""
    torch.set_float32_matmul_precision("high")
    options = {"max_autotune": True, "triton.cudagraphs": False}
    actor._prefix = torch.compile(actor._prefix, options=options)
    actor._step = torch.compile(actor._step, options=options)


def to_env_action(action, post, env_post):
    return env_post({ACTION: post(action)})[ACTION]


def worker_spec(value: str) -> tuple[str, int]:
    try:
        scope, count = value.split("=", 1)
        count = int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected SCOPE=COUNT.") from error
    if not scope or count < 0:
        raise argparse.ArgumentTypeError("Worker counts must be non-negative.")
    return scope, count


def positive(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return count


def successes(info, count: int) -> torch.Tensor:
    values = info.get("final_info", info.get("is_success", [False] * count))
    if isinstance(values, dict):
        values = values.get("is_success", [False] * count)
    if not hasattr(values, "__len__") or isinstance(values, str):
        values = [values] * count
    return torch.tensor(
        [bool(value.get("is_success", False) if isinstance(value, dict) else value) for value in values]
    )


def collect_group(suite, tasks, workers, group_offset, worker_offset, actor, critic, pre, env_pre, post, env_post, cfg, device, progress):
    """Collect fixed-length action chunks, as in RLinf's LIBERO rollout."""
    offsets = [0]
    for worker_count in workers:
        offsets.append(offsets[-1] + worker_count)
    count = offsets[-1]
    if horizon(suite) % cfg.action_steps:
        raise ValueError("The suite horizon must be divisible by action_steps for fixed-chunk rollout.")
    env = make_env(
        tasks,
        workers,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
    )
    buffer, results = RolloutBuffer(), [TaskResult(task) for task in tasks]

    try:
        observation, _ = env.reset()
        language = env.call("task_description")
        elapsed = 0
        finished = torch.zeros(count, dtype=torch.bool)
        previous_step_reward = torch.zeros(count)

        while elapsed < horizon(suite):
            data = batch(observation, language, pre, env_pre, device)
            with torch.no_grad(), amp(cfg):
                sample = actor(data)
                old_values = critic(sample.critic_features).float()

            reward = torch.zeros(count)
            done = torch.zeros(count, dtype=torch.bool)
            steps = torch.full((count,), cfg.action_steps, dtype=torch.long)

            for action in sample.actions.unbind(1):
                command = to_env_action(action, post, env_post).cpu().numpy()
                observation, step_reward, terminated, truncated, info = env.step(command)

                terminal = torch.as_tensor(terminated | truncated)
                # RLinf's LIBERO reward is the change in its binary terminal
                # reward, not the raw reward returned by robosuite/LIBERO.
                # Keep that reward state fixed after a worker finishes: the
                # remaining actions of this fixed chunk belong to an episode
                # that has already ended and must not turn its +1 into -1.
                step_reward = torch.as_tensor(terminated, dtype=reward.dtype)
                step_reward = torch.where(finished, previous_step_reward, step_reward)
                reward += step_reward - previous_step_reward
                previous_step_reward = step_reward
                done |= terminal
                elapsed += 1

                completed = terminal & ~finished
                success = successes(info, count) & completed
                for task_id, result in enumerate(results):
                    indices = slice(offsets[task_id], offsets[task_id + 1])
                    task_completed = completed[indices]
                    result.successes += int(success[indices].sum())
                    result.failures += int(task_completed.sum() - success[indices].sum())
                finished |= terminal

            # RLinf runs every action in a chunk with auto-reset disabled, then
            # resets only workers that terminated before the next policy call.
            if done.any():
                observation, _ = env.reset(options={"reset_mask": done.numpy()})
                finished[done] = False
                previous_step_reward[done] = 0

            for task_id, (start, stop) in enumerate(zip(offsets, offsets[1:])):
                for index in range(start, stop):
                    group = group_offset + task_id
                    buffer.items.append(
                        Transition(
                            group,
                            worker_offset + index,
                            {key: value[index : index + 1].cpu() for key, value in data.items()},
                            sample.path[index : index + 1].cpu(),
                            executed_log_prob(sample.log_prob[index : index + 1], steps[index : index + 1]).cpu(),
                            old_values[index : index + 1].cpu(),
                            sample.noise_std[index].item(),
                            float(reward[index]),
                            bool(done[index]),
                            int(steps[index]),
                        )
                    )
            progress.update(count)

        with torch.no_grad(), amp(cfg):
            bootstrap_sample = actor(batch(observation, language, pre, env_pre, device))
            bootstrap_values = critic(bootstrap_sample.critic_features).float().cpu()

        for index in range(count):
            buffer.bootstrap[worker_offset + index] = bootstrap_values[index : index + 1]

        return buffer, results, count
    finally:
        env.close()


def collect_suite(suite, group_offset, worker_offset, actor, critic, pre, env_pre, post, env_post, cfg, device):
    """Collect a suite in bounded environment groups, then merge its PPO data."""
    tasks = suite_tasks(suite)
    tasks, workers = active_tasks(tasks, worker_counts(tasks, cfg.workers))
    if not tasks:
        raise ValueError(f"{suite} has no active workers.")

    buffer, results, worker_count = RolloutBuffer(), [], 0
    progress = tqdm(total=sum(workers) * horizon(suite) // cfg.action_steps, desc="collecting", unit="chunk")
    try:
        for task_group, worker_group in task_groups(tasks, workers, cfg.max_concurrent_envs):
            rollout, group_results, count = collect_group(
                suite,
                task_group,
                worker_group,
                group_offset + len(results),
                worker_offset + worker_count,
                actor,
                critic,
                pre,
                env_pre,
                post,
                env_post,
                cfg,
                device,
                progress,
            )
            buffer.extend(rollout)
            results.extend(group_results)
            worker_count += count
        return buffer, results, worker_count
    finally:
        progress.close()


def update(actor, critic, actor_opt, critic_opt, buffer, cfg, device):
    data = buffer.tensors(device)
    with torch.no_grad():
        values = data["old_value"].float()
        bootstrap = {worker: value.to(device).float() for worker, value in buffer.bootstrap.items()}

    advantage, target = gae(
        data["reward"], data["done"], values, data["worker"], bootstrap, cfg.gamma, cfg.gae_lambda
    )
    advantage = normalize_advantage(advantage, data["task"])

    losses = torch.zeros(3, device=device)
    clipped = torch.zeros(int(data["task"].max()) + 1, device=device)
    samples = torch.zeros_like(clipped)
    loss_count = 0

    epoch_stats = []
    progress = tqdm(total=cfg.ppo_epochs, desc="updating", unit="epoch")
    try:
        for _ in range(cfg.ppo_epochs):
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            sample_count = len(buffer.items) // cfg.minibatch * cfg.minibatch

            epoch_logratios = []
            epoch_clipped = torch.zeros((), device=device)
            epoch_samples = 0
            for batch_index, indices in enumerate(
                torch.randperm(len(buffer.items), device=device)[:sample_count].split(cfg.minibatch)
            ):
                batches = [buffer.items[index].batch for index in indices.cpu().tolist()]
                minibatch = {key: value.to(device) for key, value in merge(batches).items()}
                old_log_prob = data["old_log_prob"][indices].float()
                with amp(cfg):
                    new_log_prob, entropy, critic_features = actor(minibatch, data["path"][indices], return_entropy=True)
                    new_log_prob = executed_log_prob(
                        new_log_prob,
                        data["steps"][indices],
                    ).float()
                    terms = loss(
                        new_log_prob,
                        old_log_prob,
                        advantage[indices],
                        critic(critic_features).float(),
                        values[indices],
                        target[indices],
                        cfg.clip,
                        cfg.value_clip,
                        cfg.huber_delta,
                        entropy.float(),
                        cfg.entropy_bonus,
                    )

                # PPO diagnostics use the same log-probabilities that enter the loss.
                with torch.no_grad():
                    epoch_logratios.append((new_log_prob - old_log_prob).detach())
                    epoch_clipped += terms.clipped.float().sum()
                    epoch_samples += terms.clipped.numel()
                losses += torch.stack((terms.policy.detach(), terms.value.detach(), terms.entropy.detach()))
                tasks = data["task"][indices]
                clipped += torch.bincount(tasks, terms.clipped.float(), minlength=len(clipped))
                samples += torch.bincount(tasks, minlength=len(samples))
                loss_count += 1

                ((terms.policy + terms.value) / cfg.gradient_accumulation_steps).backward()
                if (batch_index + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(actor_opt.param_groups[0]["params"], cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(critic_opt.param_groups[0]["params"], cfg.max_grad_norm)
                    actor_opt.step()
                    critic_opt.step()
                    actor_opt.zero_grad()
                    critic_opt.zero_grad()

            logratio = torch.cat(epoch_logratios).float()
            approx_kl = ((logratio.exp() - 1.0) - logratio).mean().item()
            clip_fraction = (epoch_clipped / epoch_samples).item()
            epoch_stats.append({"approx_kl": approx_kl, "clip_fraction": clip_fraction})
            progress.update()
    finally:
        progress.close()

    clips = [(int(count), int(total)) for count, total in zip(clipped.tolist(), samples.tolist())]
    task_noise_stds = [data["noise_std"][data["task"] == task].mean().item() for task in range(len(clipped))]
    return (
        data,
        dict(zip(("actor_loss", "critic_loss", "entropy"), (losses / loss_count).tolist())),
        clips,
        task_noise_stds,
        epoch_stats,
    )


def main(cfg) -> None:
    cfg.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    policy = SmolVLAPolicy.from_pretrained(cfg.checkpoint).to(device).eval()
    policy.config.num_steps = cfg.denoise_steps
    policy.config.train_expert_only = cfg.train_expert_only
    policy.config.train_state_proj = cfg.train_state_proj
    policy.config.freeze_vision_encoder = cfg.freeze_vision_encoder
    policy.config.pad_language_to = cfg.pad_language_to
    policy.config.tokenizer_max_length = cfg.tokenizer_max_length

    actor = FlowNoiseActor(policy, cfg.action_steps).to(device)
    critic = ValueCritic(policy.model.vlm_with_expert.config.text_config.hidden_size).to(device)
    if cfg.compile_model:
        compile_actor(actor)

    betas = (cfg.adam_beta1, cfg.adam_beta2)
    actor_opt = torch.optim.AdamW(trainable_parameters(actor), lr=cfg.actor_lr, betas=betas)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_lr, betas=betas)
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(cfg.checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {
                "padding": cfg.pad_language_to,
                "max_length": cfg.tokenizer_max_length,
            },
        },
    )
    env_pre, env_post = make_env_pre_post_processors(LiberoEnv(), policy.config)
    schedule = SuiteSchedule(cfg.suites, cfg.suites_per_update)

    with (cfg.output / "log.txt").open("a", encoding="utf-8") as log_file:
        for update_index in range(cfg.updates):
            buffer, results, group_offset, worker_offset = RolloutBuffer(), [], 0, 0
            for suite in schedule.next():
                rollout, suite_results, worker_count = collect_suite(
                    suite, group_offset, worker_offset, actor, critic, pre, env_pre, post, env_post, cfg, device
                )
                buffer.extend(rollout)
                results.extend(suite_results)
                group_offset += len(suite_results)
                worker_offset += worker_count

            data, losses, clips, task_noise_stds, epoch_stats = update(
                actor, critic, actor_opt, critic_opt, buffer, cfg, device
            )
            successes_total = sum(result.successes for result in results)
            episodes_total = sum(result.episodes for result in results)
            log(
                f"update={update_index + 1} | reward={data['reward'].mean().item():.3f} | "
                f"success_rate={successes_total}/{episodes_total} | "
                f"actor_loss={losses['actor_loss']:.4f} | critic_loss={losses['critic_loss']:.4f} | "
                f"entropy={losses['entropy']:.4f}",
                log_file,
            )
            for epoch_index, stats in enumerate(epoch_stats, start=1):
                log(
                    f"  ppo_epoch={epoch_index} | approx_kl={stats['approx_kl']:.6f} "
                    f"| clip_fraction={stats['clip_fraction']:.3f}",
                    log_file,
                )
            for result, (clipped, total), noise_std in zip(results, clips, task_noise_stds):
                log(
                    f"  - {result.label} | clip_fraction={clipped / total:.3f} "
                    f"| noise_std={noise_std:.6f}",
                    log_file,
                )

            if (update_index + 1) % 10 == 0 or update_index + 1 == cfg.updates:
                directory = cfg.output / f"update_{update_index + 1:05d}" / "pretrained_model"
                directory.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(directory)
                pre.save_pretrained(directory)
                post.save_pretrained(directory)
                torch.save(
                    {"noise": actor.noise.state_dict(), "critic": critic.state_dict(), "update": update_index + 1},
                    directory.parent / "rl_state.pt",
                )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/flow_noise"))
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=["libero_spatial"])
    parser.add_argument("--suites-per-update", type=int, default=1)
    parser.add_argument("--workers", action="extend", type=worker_spec, nargs="+", metavar="SCOPE=COUNT")
    parser.add_argument("--max-concurrent-envs", type=positive, default=24)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--action-steps", type=int, default=10)
    parser.add_argument("--denoise-steps", type=int, default=5)
    parser.add_argument("--pad-language-to", choices=("longest", "max_length"), default="max_length")
    parser.add_argument("--tokenizer-max-length", type=positive, default=48)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=positive, default=4)
    parser.add_argument("--actor-lr", type=float, default=5e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--entropy-bonus", type=float, default=0.005)
    parser.add_argument("--fp32", action="store_false", dest="bfloat16")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--train-expert-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-state-proj", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=True)
    cfg = parser.parse_args(argv)
    cfg.workers = {"default": 2, **dict(cfg.workers or [])}
    return cfg


if __name__ == "__main__":
    main(parse_args())
