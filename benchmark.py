from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.agents import HybridAgent, RuleAgent
from src.environments import KeyDoorEnvironment
from src.learning import LearningController
from src.memory import ExperienceBuffer
from src.predicates import PredicateGenerator
from src.rl import QTablePolicy, SARSAPolicy
from src.rules import RuleMiner, RuleStore


# ============================================================
# РЕЗУЛЬТАТ ОДНОГО ЭПИЗОДА
# ============================================================


@dataclass
class BenchmarkEpisode:
    episode: int
    total_reward: float
    steps: int
    success: bool
    final_event: str
    epsilon: float
    rule_count: int
    state_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# РЕЗУЛЬТАТ ЭКСПЕРИМЕНТА
# ============================================================


@dataclass
class BenchmarkResult:
    name: str
    episodes: list[BenchmarkEpisode]

    @property
    def rewards(self) -> list[float]:
        return [
            episode.total_reward
            for episode in self.episodes
        ]

    @property
    def steps(self) -> list[int]:
        return [
            episode.steps
            for episode in self.episodes
        ]

    @property
    def successes(self) -> list[bool]:
        return [
            episode.success
            for episode in self.episodes
        ]

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0

        return sum(self.successes) / len(self.episodes)

    @property
    def average_reward(self) -> float:
        if not self.episodes:
            return 0.0

        return mean(self.rewards)

    @property
    def average_steps(self) -> float:
        if not self.episodes:
            return 0.0

        return mean(self.steps)

    @property
    def final_success_rate(self) -> float:
        """
        Успешность на последних 100 эпизодах.
        """

        if not self.episodes:
            return 0.0

        recent = self.episodes[-100:]

        return (
            sum(episode.success for episode in recent)
            / len(recent)
        )

    @property
    def final_average_reward(self) -> float:
        """
        Средняя награда на последних 100 эпизодах.
        """

        if not self.episodes:
            return 0.0

        recent = self.episodes[-100:]

        return mean(
            episode.total_reward
            for episode in recent
        )

    def summary(self) -> dict[str, Any]:
        if not self.episodes:
            final_rule_count = 0
            final_state_count = 0
        else:
            final_rule_count = self.episodes[-1].rule_count
            final_state_count = self.episodes[-1].state_count

        return {
            "name": self.name,
            "episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "final_100_success_rate": (
                self.final_success_rate
            ),
            "final_100_average_reward": (
                self.final_average_reward
            ),
            "final_rule_count": final_rule_count,
            "final_state_count": final_state_count,
        }


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================


def rolling_average(
    values: list[float],
    window: int,
) -> list[float]:
    if window <= 0:
        raise ValueError(
            "window must be greater than zero"
        )

    result: list[float] = []

    for index in range(len(values)):
        start = max(
            0,
            index - window + 1,
        )

        current_values = values[start:index + 1]

        result.append(
            sum(current_values)
            / len(current_values)
        )

    return result


def create_environment() -> KeyDoorEnvironment:
    return KeyDoorEnvironment(
        width=7,
        height=7,
        max_steps=100,
    )


def print_progress(
    experiment_name: str,
    episode: int,
    total_episodes: int,
    episodes: list[BenchmarkEpisode],
    log_interval: int,
) -> None:
    if (
        episode % log_interval != 0
        and episode != total_episodes
    ):
        return

    recent = episodes[
        max(0, len(episodes) - log_interval):
    ]

    recent_reward = mean(
        item.total_reward
        for item in recent
    )

    recent_success = (
        sum(item.success for item in recent)
        / len(recent)
    )

    print(
        f"[{experiment_name}] "
        f"episode={episode}/{total_episodes} "
        f"reward={recent_reward:.3f} "
        f"success={recent_success:.1%}"
    )


# ============================================================
# Q-LEARNING
# ============================================================


def run_q_learning(
    episodes: int,
    seed: int,
    log_interval: int,
) -> BenchmarkResult:
    environment = create_environment()

    policy = QTablePolicy(
        actions=environment.ACTIONS,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        seed=seed,
    )

    history: list[BenchmarkEpisode] = []

    for episode_number in range(1, episodes + 1):
        state = environment.reset()

        total_reward = 0.0
        final_event = "unknown"

        while True:
            action = policy.choose_action(
                state,
                explore=True,
            )

            step_result = environment.step(action)

            policy.update(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.state,
                done=step_result.done,
            )

            total_reward += step_result.reward

            final_event = str(
                step_result.info.get(
                    "event",
                    "unknown",
                )
            )

            state = step_result.state

            if step_result.done:
                break

        history.append(
            BenchmarkEpisode(
                episode=episode_number,
                total_reward=total_reward,
                steps=environment.step_count,
                success=step_result.success,
                final_event=final_event,
                epsilon=policy.epsilon,
                rule_count=0,
                state_count=policy.state_count,
            )
        )

        policy.decay_epsilon(
            decay_rate=0.995,
            minimum_epsilon=0.05,
        )

        print_progress(
            experiment_name="Q-learning",
            episode=episode_number,
            total_episodes=episodes,
            episodes=history,
            log_interval=log_interval,
        )

    return BenchmarkResult(
        name="Q-learning",
        episodes=history,
    )


# ============================================================
# SARSA
# ============================================================


def run_sarsa(
    episodes: int,
    seed: int,
    log_interval: int,
) -> BenchmarkResult:
    environment = create_environment()

    policy = SARSAPolicy(
        actions=environment.ACTIONS,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        seed=seed,
    )

    history: list[BenchmarkEpisode] = []

    for episode_number in range(1, episodes + 1):
        state = environment.reset()

        action = policy.choose_action(
            state,
            explore=True,
        )

        total_reward = 0.0
        final_event = "unknown"

        while True:
            step_result = environment.step(action)

            total_reward += step_result.reward

            final_event = str(
                step_result.info.get(
                    "event",
                    "unknown",
                )
            )

            if step_result.done:
                policy.update(
                    state=state,
                    action=action,
                    reward=step_result.reward,
                    next_state=step_result.state,
                    next_action=None,
                    done=True,
                )

                break

            next_action = policy.choose_action(
                step_result.state,
                explore=True,
            )

            policy.update(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.state,
                next_action=next_action,
                done=False,
            )

            state = step_result.state
            action = next_action

        history.append(
            BenchmarkEpisode(
                episode=episode_number,
                total_reward=total_reward,
                steps=environment.step_count,
                success=step_result.success,
                final_event=final_event,
                epsilon=policy.epsilon,
                rule_count=0,
                state_count=policy.state_count,
            )
        )

        policy.decay_epsilon(
            decay_rate=0.995,
            minimum_epsilon=0.05,
        )

        print_progress(
            experiment_name="SARSA",
            episode=episode_number,
            total_episodes=episodes,
            episodes=history,
            log_interval=log_interval,
        )

    return BenchmarkResult(
        name="SARSA",
        episodes=history,
    )


# ============================================================
# HYBRID AGENT
# ============================================================


def run_hybrid(
    episodes: int,
    seed: int,
    log_interval: int,
) -> tuple[
    BenchmarkResult,
    RuleAgent,
    RuleStore,
]:
    environment = create_environment()

    predicate_generator = PredicateGenerator()

    rule_store = RuleStore()

    rule_miner = RuleMiner(
        predicate_generator=predicate_generator,
        minimum_reward=0.0,
    )

    experience_buffer = ExperienceBuffer(
        capacity=50000,
        seed=seed,
    )

    q_policy = QTablePolicy(
        actions=environment.ACTIONS,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        seed=seed,
    )

    rule_agent = RuleAgent(
        rule_store=rule_store,
        predicate_generator=predicate_generator,
    )

    hybrid_agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=q_policy,
    )

    controller = LearningController(
        agent=hybrid_agent,
        q_policy=q_policy,
        experience_buffer=experience_buffer,
        rule_miner=rule_miner,
        rule_store=rule_store,
    )

    history: list[BenchmarkEpisode] = []

    for episode_number in range(1, episodes + 1):
        state = environment.reset()

        hybrid_agent.reset_decision_history()

        total_reward = 0.0
        final_event = "unknown"

        while True:
            action = controller.choose_action(state)

            decision_source = hybrid_agent.last_source

            step_result = environment.step(action)

            controller.learn(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.state,
                done=step_result.done,
                success=step_result.success,
                source=decision_source,
            )

            total_reward += step_result.reward

            final_event = str(
                step_result.info.get(
                    "event",
                    "unknown",
                )
            )

            state = step_result.state

            if step_result.done:
                break

        history.append(
            BenchmarkEpisode(
                episode=episode_number,
                total_reward=total_reward,
                steps=environment.step_count,
                success=step_result.success,
                final_event=final_event,
                epsilon=q_policy.epsilon,
                rule_count=len(rule_store.rules),
                state_count=q_policy.state_count,
            )
        )

        controller.decay_epsilon(
            decay_rate=0.995,
            minimum_epsilon=0.05,
        )

        print_progress(
            experiment_name="Hybrid Agent",
            episode=episode_number,
            total_episodes=episodes,
            episodes=history,
            log_interval=log_interval,
        )

    return (
        BenchmarkResult(
            name="Hybrid Agent",
            episodes=history,
        ),
        rule_agent,
        rule_store,
    )


# ============================================================
# RULE AGENT
# ============================================================


def evaluate_rule_agent(
    rule_agent: RuleAgent,
    rule_store: RuleStore,
    episodes: int,
    seed: int,
    log_interval: int,
) -> BenchmarkResult:
    """
    Проверяет только правила, найденные Hybrid Agent.

    Если подходящего правила нет, используется случайное
    действие. Поэтому в графиках этот агент обозначается
    как Rule Agent.
    """

    environment = create_environment()
    random_generator = random.Random(seed)

    history: list[BenchmarkEpisode] = []

    for episode_number in range(1, episodes + 1):
        state = environment.reset()

        rule_agent.reset_decision_history()

        total_reward = 0.0
        final_event = "unknown"

        while True:
            action = rule_agent.choose_action(state)

            if action is None:
                action = random_generator.choice(
                    environment.ACTIONS
                )

            step_result = environment.step(action)

            total_reward += step_result.reward

            final_event = str(
                step_result.info.get(
                    "event",
                    "unknown",
                )
            )

            state = step_result.state

            if step_result.done:
                break

        history.append(
            BenchmarkEpisode(
                episode=episode_number,
                total_reward=total_reward,
                steps=environment.step_count,
                success=step_result.success,
                final_event=final_event,
                epsilon=0.0,
                rule_count=len(rule_store.rules),
                state_count=0,
            )
        )

        print_progress(
            experiment_name="Rule Agent",
            episode=episode_number,
            total_episodes=episodes,
            episodes=history,
            log_interval=log_interval,
        )

    return BenchmarkResult(
        name="Rule Agent",
        episodes=history,
    )


# ============================================================
# ЭКСПОРТ CSV И JSON
# ============================================================


def export_history(
    result: BenchmarkResult,
    output_directory: Path,
) -> None:
    experiment_directory = (
        output_directory
        / result.name.lower().replace(" ", "_")
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        experiment_directory
        / "training_history.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "episode",
            "total_reward",
            "steps",
            "success",
            "final_event",
            "epsilon",
            "rule_count",
            "state_count",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for episode in result.episodes:
            writer.writerow(
                episode.to_dict()
            )

    json_path = (
        experiment_directory
        / "summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.summary(),
            file,
            ensure_ascii=False,
            indent=2,
        )


def export_summary(
    results: list[BenchmarkResult],
    output_directory: Path,
) -> None:
    summaries = [
        result.summary()
        for result in results
    ]

    json_path = (
        output_directory
        / "benchmark_summary.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summaries,
            file,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = (
        output_directory
        / "benchmark_summary.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        if not summaries:
            return

        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0]),
        )

        writer.writeheader()
        writer.writerows(summaries)


# ============================================================
# ГРАФИКИ
# ============================================================


def save_comparison_plot(
    results: list[BenchmarkResult],
    output_path: Path,
    metric: str,
    title: str,
    y_label: str,
    rolling_window: int,
) -> None:
    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for result in results:
        episode_numbers = [
            episode.episode
            for episode in result.episodes
        ]

        if metric == "reward":
            values = [
                episode.total_reward
                for episode in result.episodes
            ]

        elif metric == "success":
            values = [
                float(episode.success)
                for episode in result.episodes
            ]

        elif metric == "steps":
            values = [
                float(episode.steps)
                for episode in result.episodes
            ]

        else:
            raise ValueError(
                f"unknown metric: {metric}"
            )

        smoothed_values = rolling_average(
            values,
            window=rolling_window,
        )

        axis.plot(
            episode_numbers,
            smoothed_values,
            label=result.name,
        )

    axis.set_title(title)
    axis.set_xlabel("Episode")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def save_summary_bar_plot(
    results: list[BenchmarkResult],
    output_path: Path,
) -> None:
    names = [
        result.name
        for result in results
    ]

    success_rates = [
        result.final_success_rate * 100.0
        for result in results
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    bars = axis.bar(
        names,
        success_rates,
    )

    axis.set_title(
        "Success rate during final 100 episodes"
    )
    axis.set_ylabel("Success rate, %")
    axis.set_ylim(0, 100)
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        success_rates,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def create_plots(
    results: list[BenchmarkResult],
    output_directory: Path,
    rolling_window: int,
) -> None:
    plots_directory = (
        output_directory / "plots"
    )

    plots_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_comparison_plot(
        results=results,
        output_path=(
            plots_directory
            / "reward_comparison.png"
        ),
        metric="reward",
        title=(
            f"Average reward "
            f"(rolling window {rolling_window})"
        ),
        y_label="Reward",
        rolling_window=rolling_window,
    )

    save_comparison_plot(
        results=results,
        output_path=(
            plots_directory
            / "success_rate_comparison.png"
        ),
        metric="success",
        title=(
            f"Success rate "
            f"(rolling window {rolling_window})"
        ),
        y_label="Success rate",
        rolling_window=rolling_window,
    )

    save_comparison_plot(
        results=results,
        output_path=(
            plots_directory
            / "steps_comparison.png"
        ),
        metric="steps",
        title=(
            f"Episode steps "
            f"(rolling window {rolling_window})"
        ),
        y_label="Steps",
        rolling_window=rolling_window,
    )

    save_summary_bar_plot(
        results=results,
        output_path=(
            plots_directory
            / "final_success_comparison.png"
        ),
    )


# ============================================================
# ТАБЛИЦА В ТЕРМИНАЛЕ
# ============================================================


def print_summary_table(
    results: list[BenchmarkResult],
) -> None:
    print()
    print("=" * 96)
    print("BENCHMARK SUMMARY")
    print("=" * 96)

    header = (
        f"{'Agent':<20}"
        f"{'Success':>12}"
        f"{'Last 100':>12}"
        f"{'Avg reward':>15}"
        f"{'Avg steps':>12}"
        f"{'Rules':>10}"
        f"{'States':>10}"
    )

    print(header)
    print("-" * 96)

    for result in results:
        summary = result.summary()

        print(
            f"{result.name:<20}"
            f"{summary['success_rate']:>11.1%}"
            f"{summary['final_100_success_rate']:>11.1%}"
            f"{summary['average_reward']:>15.3f}"
            f"{summary['average_steps']:>12.1f}"
            f"{summary['final_rule_count']:>10}"
            f"{summary['final_state_count']:>10}"
        )

    print("=" * 96)


# ============================================================
# MAIN
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Hybrid Agent, Q-learning, "
            "SARSA and Rule Agent."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help=(
            "Number of episodes for every "
            "training experiment."
        ),
    )

    parser.add_argument(
        "--rule-evaluation-episodes",
        type=int,
        default=500,
        help=(
            "Number of episodes used to evaluate "
            "the learned Rule Agent."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Rolling average window.",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Terminal logging interval.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmark"),
        help="Output directory.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.episodes <= 0:
        raise ValueError(
            "episodes must be greater than zero"
        )

    if arguments.rule_evaluation_episodes <= 0:
        raise ValueError(
            "rule evaluation episodes must be "
            "greater than zero"
        )

    output_directory: Path = arguments.output

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Starting benchmark")
    print(
        f"Training episodes: {arguments.episodes}"
    )
    print(
        "Rule evaluation episodes: "
        f"{arguments.rule_evaluation_episodes}"
    )
    print(f"Seed: {arguments.seed}")
    print()

    hybrid_result, rule_agent, rule_store = (
        run_hybrid(
            episodes=arguments.episodes,
            seed=arguments.seed,
            log_interval=arguments.log_interval,
        )
    )

    q_learning_result = run_q_learning(
        episodes=arguments.episodes,
        seed=arguments.seed,
        log_interval=arguments.log_interval,
    )

    sarsa_result = run_sarsa(
        episodes=arguments.episodes,
        seed=arguments.seed,
        log_interval=arguments.log_interval,
    )

    rule_result = evaluate_rule_agent(
        rule_agent=rule_agent,
        rule_store=rule_store,
        episodes=(
            arguments.rule_evaluation_episodes
        ),
        seed=arguments.seed,
        log_interval=arguments.log_interval,
    )

    results = [
        hybrid_result,
        q_learning_result,
        sarsa_result,
        rule_result,
    ]

    for result in results:
        export_history(
            result=result,
            output_directory=output_directory,
        )

    export_summary(
        results=results,
        output_directory=output_directory,
    )

    create_plots(
        results=results,
        output_directory=output_directory,
        rolling_window=arguments.rolling_window,
    )

    print_summary_table(results)

    print()
    print("Benchmark completed.")
    print(
        f"Results saved to: "
        f"{output_directory.resolve()}"
    )

    print()
    print("Generated plots:")

    for path in sorted(
        (output_directory / "plots").glob("*.png")
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()