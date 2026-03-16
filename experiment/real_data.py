"""
Real data loading, parameter fitting, and replay-based evaluation.

Supports two datasets:
  1. Duolingo Spaced Repetition (Settles & Meeder 2016) — 13M learning traces
  2. ASSISTments 2012-2013 — 2.5M math tutoring interactions

Evaluation approaches:
  A. Fitted-parameter simulation: Fit student model params from real data,
     then run synthetic experiment with realistic parameters.
  B. Replay evaluation: Replay logged student trajectories under different
     bandit policies using the fitted model.
"""

import math
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import ExperimentConfig
from .selectors import BaseSelector, OracleSelector, create_selector, SELECTOR_REGISTRY
from .simulation import GroupMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    """A single student-skill interaction from real data."""
    student_id: str
    skill_id: int
    correct: bool
    timestamp: float  # seconds since epoch (or relative)
    delta: float = 0.0  # seconds since last exposure to this skill


@dataclass
class StudentTrace:
    """A student's complete interaction trace, sorted by time."""
    student_id: str
    interactions: List[Interaction]
    skills: List[int]  # unique skill IDs this student encountered


@dataclass
class RealDataset:
    """Preprocessed dataset ready for parameter fitting and replay."""
    name: str
    traces: List[StudentTrace]
    num_students: int
    num_skills: int
    num_interactions: int
    skill_names: Dict[int, str]  # skill_id -> human-readable name


@dataclass
class FittedParams:
    """Student model parameters fitted from real data."""
    learning_rate: float  # alpha
    incorrect_penalty: float  # beta
    decay_rate: float  # lambda (per-second, converted to per-timestep)
    base_knowledge: float  # floor after forgetting
    initial_knowledge_mean: float
    fit_loss: float  # final loss value
    dataset_name: str
    num_students_used: int
    num_interactions_used: int

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "incorrect_penalty": self.incorrect_penalty,
            "decay_rate": self.decay_rate,
            "base_knowledge": self.base_knowledge,
            "initial_knowledge_mean": self.initial_knowledge_mean,
            "fit_loss": self.fit_loss,
            "dataset_name": self.dataset_name,
            "num_students_used": self.num_students_used,
            "num_interactions_used": self.num_interactions_used,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FittedParams":
        with open(path) as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_duolingo(data_dir: str, max_students: int = 1000,
                  min_interactions: int = 50,
                  language_track: str = "en_es") -> RealDataset:
    """
    Load Duolingo Spaced Repetition dataset.

    The dataset has columns:
      p_recall, timestamp, delta, user_id, learning_language, ui_language,
      lexeme_id, lexeme_string, history_seen, history_correct,
      session_seen, session_correct

    We group by (user_id, lexeme_id) pairs where lexeme_id maps to skills.

    Args:
        data_dir: Path to data/duolingo/ directory
        max_students: Maximum number of students to load
        min_interactions: Minimum interactions per student to include
        language_track: Language track to filter (e.g., "en_es")
    """
    data_path = Path(data_dir)

    # Try multiple possible filenames
    candidates = [
        data_path / "settles.acl16.learning_traces.13m.csv",
        data_path / "learning_traces.13m.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"Duolingo data not found in {data_dir}. "
            f"Run: bash scripts/download_data.sh duolingo"
        )

    logger.info(f"Loading Duolingo data from {csv_path}...")

    # Read CSV — it's large, so use chunked reading
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=500_000):
        # Filter by language track
        if language_track:
            chunk = chunk[chunk["learning_language"] == language_track.split("_")[0]]
            if "ui_language" in chunk.columns:
                chunk = chunk[chunk["ui_language"] == language_track.split("_")[1]]
        chunks.append(chunk)
        # Early stop if we have enough data
        total_rows = sum(len(c) for c in chunks)
        if total_rows > max_students * 500:
            break

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df)} rows after language filter")

    return _build_dataset_from_df(
        df=df,
        name="duolingo",
        student_col="user_id",
        skill_col="lexeme_id",
        correct_col="p_recall",  # probability; we threshold at 0.5
        timestamp_col="timestamp",
        delta_col="delta",
        correct_is_probability=True,
        max_students=max_students,
        min_interactions=min_interactions,
    )


def load_assistments(data_dir: str, max_students: int = 1000,
                     min_interactions: int = 50) -> RealDataset:
    """
    Load ASSISTments 2012-2013 dataset.

    Key columns: user_id, skill_id, correct, start_time

    Args:
        data_dir: Path to data/assistments/ directory
        max_students: Maximum number of students to load
        min_interactions: Minimum interactions per student to include
    """
    data_path = Path(data_dir)

    candidates = [
        data_path / "2012-2013-data-with-predictions-4-final.csv",
        data_path / "skill_builder_data.csv",
    ]
    csv_path = None
    for c in candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"ASSISTments data not found in {data_dir}. "
            f"Run: bash scripts/download_data.sh assistments"
        )

    logger.info(f"Loading ASSISTments data from {csv_path}...")

    # Determine which columns exist
    sample = pd.read_csv(csv_path, nrows=5)
    cols = sample.columns.tolist()

    # Map to canonical names
    student_col = "user_id" if "user_id" in cols else "student_id"
    skill_col = "skill_id" if "skill_id" in cols else "problem_id"
    correct_col = "correct"
    timestamp_col = None
    for tc in ["start_time", "problem_start_time", "order_id"]:
        if tc in cols:
            timestamp_col = tc
            break

    usecols = [c for c in [student_col, skill_col, correct_col, timestamp_col] if c]
    df = pd.read_csv(csv_path, usecols=usecols)

    # Drop rows with missing skill
    df = df.dropna(subset=[skill_col])

    # If no timestamp, use row order as proxy
    if timestamp_col is None or timestamp_col == "order_id":
        df["_timestamp"] = range(len(df))
        timestamp_col = "_timestamp"

    logger.info(f"Loaded {len(df)} rows")

    return _build_dataset_from_df(
        df=df,
        name="assistments",
        student_col=student_col,
        skill_col=skill_col,
        correct_col=correct_col,
        timestamp_col=timestamp_col,
        delta_col=None,
        correct_is_probability=False,
        max_students=max_students,
        min_interactions=min_interactions,
    )


def _build_dataset_from_df(
    df: pd.DataFrame,
    name: str,
    student_col: str,
    skill_col: str,
    correct_col: str,
    timestamp_col: str,
    delta_col: Optional[str],
    correct_is_probability: bool,
    max_students: int,
    min_interactions: int,
) -> RealDataset:
    """Build a RealDataset from a preprocessed DataFrame."""

    # Map skill IDs to consecutive integers
    unique_skills = df[skill_col].unique()
    skill_to_idx = {s: i for i, s in enumerate(unique_skills)}
    skill_names = {i: str(s) for s, i in skill_to_idx.items()}

    # Filter students by interaction count
    student_counts = df[student_col].value_counts()
    valid_students = student_counts[student_counts >= min_interactions].index
    if len(valid_students) > max_students:
        valid_students = valid_students[:max_students]

    df = df[df[student_col].isin(valid_students)]

    # Sort by student and timestamp
    df = df.sort_values([student_col, timestamp_col])

    # Build traces
    traces = []
    total_interactions = 0

    for student_id, group in df.groupby(student_col):
        interactions = []
        last_skill_time: Dict[int, float] = {}

        for _, row in group.iterrows():
            skill_idx = skill_to_idx[row[skill_col]]
            ts = float(row[timestamp_col])

            if correct_is_probability:
                correct = float(row[correct_col]) >= 0.5
            else:
                correct = bool(row[correct_col])

            # Compute delta (time since last exposure to this skill)
            if delta_col and pd.notna(row.get(delta_col)):
                delta = float(row[delta_col])
            elif skill_idx in last_skill_time:
                delta = ts - last_skill_time[skill_idx]
            else:
                delta = 0.0

            last_skill_time[skill_idx] = ts

            interactions.append(Interaction(
                student_id=str(student_id),
                skill_id=skill_idx,
                correct=correct,
                timestamp=ts,
                delta=max(0.0, delta),
            ))

        if interactions:
            unique_skills_for_student = list(set(
                ix.skill_id for ix in interactions
            ))
            traces.append(StudentTrace(
                student_id=str(student_id),
                interactions=interactions,
                skills=unique_skills_for_student,
            ))
            total_interactions += len(interactions)

    logger.info(
        f"Built {name} dataset: {len(traces)} students, "
        f"{len(skill_names)} skills, {total_interactions} interactions"
    )

    return RealDataset(
        name=name,
        traces=traces,
        num_students=len(traces),
        num_skills=len(skill_names),
        num_interactions=total_interactions,
        skill_names=skill_names,
    )


# ---------------------------------------------------------------------------
# Parameter fitting via MLE
# ---------------------------------------------------------------------------

def fit_student_model(dataset: RealDataset,
                      max_students: int = 200,
                      time_unit: float = 1.0) -> FittedParams:
    """
    Fit student model parameters (α, β, λ, base) from real data via MLE.

    The model:
      - On correct answer: k += α * (1 - k)
      - On incorrect answer: k -= β * k
      - Forgetting: k = base + (k - base) * exp(-λ * delta)
      - P(correct) = k

    We maximize log-likelihood of observed correct/incorrect outcomes.

    Args:
        dataset: Preprocessed real dataset
        max_students: Max students to use for fitting (for speed)
        time_unit: Seconds per timestep for decay rate conversion.
                   e.g., 86400 means λ is per-day.
    """
    traces = dataset.traces[:max_students]

    # Compute median inter-skill delta for time normalization
    all_deltas = []
    for trace in traces:
        for ix in trace.interactions:
            if ix.delta > 0:
                all_deltas.append(ix.delta)
    if all_deltas:
        median_delta = float(np.median(all_deltas))
        if time_unit == 1.0:
            # Auto-detect: use median delta as the "timestep"
            time_unit = max(median_delta, 1.0)
    logger.info(f"Time unit for decay: {time_unit:.1f} seconds")

    def neg_log_likelihood(params):
        alpha, beta, lam, base, k0 = params
        # Clamp to valid ranges
        alpha = np.clip(alpha, 1e-4, 0.5)
        beta = np.clip(beta, 1e-4, 0.3)
        lam = np.clip(lam, 1e-6, 1.0)
        base = np.clip(base, 0.01, 0.5)
        k0 = np.clip(k0, 0.05, 0.95)

        total_nll = 0.0
        n_obs = 0

        for trace in traces:
            # Per-skill knowledge for this student
            knowledge = {}

            for ix in trace.interactions:
                sid = ix.skill_id
                if sid not in knowledge:
                    knowledge[sid] = k0

                # Apply forgetting
                if ix.delta > 0:
                    dt = ix.delta / time_unit
                    k = knowledge[sid]
                    knowledge[sid] = base + (k - base) * math.exp(-lam * dt)

                k = np.clip(knowledge[sid], 1e-6, 1 - 1e-6)

                # Log-likelihood
                if ix.correct:
                    total_nll -= math.log(k)
                else:
                    total_nll -= math.log(1 - k)
                n_obs += 1

                # Update knowledge
                if ix.correct:
                    knowledge[sid] = k + alpha * (1 - k)
                else:
                    knowledge[sid] = k - beta * k

                knowledge[sid] = np.clip(knowledge[sid], 0.01, 0.99)

        return total_nll / max(n_obs, 1)

    # Optimize
    x0 = [0.12, 0.02, 0.01, 0.10, 0.35]
    bounds = [
        (1e-4, 0.5),   # alpha
        (1e-4, 0.3),   # beta
        (1e-6, 1.0),   # lambda
        (0.01, 0.5),   # base
        (0.05, 0.95),  # k0
    ]

    logger.info("Fitting student model parameters via MLE...")
    result = minimize(
        neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-6},
    )

    alpha, beta, lam, base, k0 = result.x
    # Convert lambda from per-time-unit to per-timestep
    # In our simulation, one timestep = one question, so lambda stays as-is
    # but we store the time_unit for reference.

    total_interactions = sum(len(t.interactions) for t in traces)

    fitted = FittedParams(
        learning_rate=float(np.clip(alpha, 1e-4, 0.5)),
        incorrect_penalty=float(np.clip(beta, 1e-4, 0.3)),
        decay_rate=float(np.clip(lam, 1e-6, 1.0)),
        base_knowledge=float(np.clip(base, 0.01, 0.5)),
        initial_knowledge_mean=float(np.clip(k0, 0.05, 0.95)),
        fit_loss=float(result.fun),
        dataset_name=dataset.name,
        num_students_used=len(traces),
        num_interactions_used=total_interactions,
    )

    logger.info(
        f"Fitted params: α={fitted.learning_rate:.4f}, β={fitted.incorrect_penalty:.4f}, "
        f"λ={fitted.decay_rate:.4f}, base={fitted.base_knowledge:.4f}, "
        f"k0={fitted.initial_knowledge_mean:.4f}, loss={fitted.fit_loss:.4f}"
    )

    return fitted


# ---------------------------------------------------------------------------
# Replay-based evaluation
# ---------------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Result of replaying a bandit policy on real student traces."""
    algorithm: str
    dataset_name: str
    num_students: int
    num_interactions: int
    # Per-student metrics
    avg_knowledge_final: float
    weakest_knowledge_final: float
    accuracy: float
    # Trajectories for plotting
    knowledge_over_time: List[float]  # average knowledge at each % of trajectory
    weakest_over_time: List[float]


def replay_evaluate(
    dataset: RealDataset,
    fitted_params: FittedParams,
    algorithms: List[str],
    max_students: int = 200,
    max_interactions_per_student: int = 500,
    num_time_bins: int = 100,
) -> List[ReplayResult]:
    """
    Evaluate bandit algorithms via replay on real student data.

    For each student trace:
      1. Use the fitted model to simulate knowledge dynamics
      2. At each step, the bandit selects which skill to quiz
      3. We look ahead in the real trace to find the next interaction
         for that skill, using its actual outcome
      4. Track knowledge evolution using the fitted model

    This is a "model-based replay": we use the fitted model to track
    knowledge but use real correctness outcomes where available.

    For skills where no real outcome is available (bandit chose a skill
    the student wasn't quizzed on next), we simulate the outcome using
    the model's P(correct) = k.

    Args:
        dataset: Preprocessed real dataset
        fitted_params: Fitted student model parameters
        algorithms: List of algorithm names to evaluate
        max_students: Max students to evaluate on
        max_interactions_per_student: Max interactions per student
        num_time_bins: Number of bins for trajectory averaging
    """
    results = []

    for algo_name in algorithms:
        logger.info(f"Replay evaluating: {algo_name}")

        all_final_knowledge = []
        all_final_weakest = []
        all_accuracy = []
        # Binned trajectories for averaging
        binned_knowledge = [[] for _ in range(num_time_bins)]
        binned_weakest = [[] for _ in range(num_time_bins)]

        total_interactions = 0

        for trace in dataset.traces[:max_students]:
            n_skills = len(trace.skills)
            if n_skills < 2:
                continue

            # Map trace skills to [0, n_skills)
            skill_map = {s: i for i, s in enumerate(trace.skills)}
            n_cats = len(skill_map)

            # Build config for this student
            config = ExperimentConfig(
                num_students_per_group=1,
                num_categories=n_cats,
                learning_rate=fitted_params.learning_rate,
                incorrect_penalty=fitted_params.incorrect_penalty,
                decay_rate=fitted_params.decay_rate,
                base_knowledge=fitted_params.base_knowledge,
                initial_knowledge_mean=fitted_params.initial_knowledge_mean,
                questions_per_session=max_interactions_per_student,
                random_seed=42,
            )

            # Create selector
            rng = np.random.default_rng(hash(trace.student_id) % (2**31))
            selector = create_selector(algo_name, config, rng=rng)

            # Knowledge state (model-based)
            knowledge = np.full(n_cats, fitted_params.initial_knowledge_mean)

            # If oracle, give it knowledge reference
            if isinstance(selector, OracleSelector):
                selector.set_knowledge_ref(knowledge)

            # Build lookup: for each skill, queue of upcoming (correct, delta)
            from collections import deque
            skill_queues: Dict[int, deque] = {i: deque() for i in range(n_cats)}
            for ix in trace.interactions:
                local_skill = skill_map[ix.skill_id]
                skill_queues[local_skill].append((ix.correct, ix.delta))

            # Track time since exposure for forgetting
            time_since_exposure = np.zeros(n_cats, dtype=np.float64)

            correct_count = 0
            total_count = 0
            trajectory_knowledge = []
            trajectory_weakest = []

            n_steps = min(max_interactions_per_student, len(trace.interactions))

            for step in range(n_steps):
                # Bandit selects a category
                cat = selector.select_category()

                # Apply forgetting to all categories
                for i in range(n_cats):
                    if time_since_exposure[i] > 0:
                        dt = time_since_exposure[i]
                        knowledge[i] = (
                            fitted_params.base_knowledge
                            + (knowledge[i] - fitted_params.base_knowledge)
                            * math.exp(-fitted_params.decay_rate * dt)
                        )
                        knowledge[i] = np.clip(knowledge[i], 0.01, 0.99)

                # Get outcome: use real data if available, else simulate
                if skill_queues[cat]:
                    real_correct, real_delta = skill_queues[cat].popleft()
                    correct = real_correct
                else:
                    # No real data for this skill at this point; simulate
                    correct = rng.random() < knowledge[cat]

                # Update knowledge
                if correct:
                    knowledge[cat] += fitted_params.learning_rate * (1 - knowledge[cat])
                    correct_count += 1
                else:
                    knowledge[cat] -= fitted_params.incorrect_penalty * knowledge[cat]
                knowledge[cat] = np.clip(knowledge[cat], 0.01, 0.99)

                # Update selector
                selector.update(cat, correct)
                total_count += 1

                # Update time tracking
                time_since_exposure += 1
                time_since_exposure[cat] = 0

                trajectory_knowledge.append(float(np.mean(knowledge)))
                trajectory_weakest.append(float(np.min(knowledge)))

            total_interactions += total_count

            if total_count > 0:
                all_final_knowledge.append(float(np.mean(knowledge)))
                all_final_weakest.append(float(np.min(knowledge)))
                all_accuracy.append(correct_count / total_count)

                # Bin trajectory for averaging
                for b in range(num_time_bins):
                    idx = int(b / num_time_bins * len(trajectory_knowledge))
                    idx = min(idx, len(trajectory_knowledge) - 1)
                    binned_knowledge[b].append(trajectory_knowledge[idx])
                    binned_weakest[b].append(trajectory_weakest[idx])

        # Average across students
        knowledge_over_time = [
            float(np.mean(bk)) if bk else 0.0 for bk in binned_knowledge
        ]
        weakest_over_time = [
            float(np.mean(bw)) if bw else 0.0 for bw in binned_weakest
        ]

        results.append(ReplayResult(
            algorithm=algo_name,
            dataset_name=dataset.name,
            num_students=len(all_final_knowledge),
            num_interactions=total_interactions,
            avg_knowledge_final=float(np.mean(all_final_knowledge)) if all_final_knowledge else 0.0,
            weakest_knowledge_final=float(np.mean(all_final_weakest)) if all_final_weakest else 0.0,
            accuracy=float(np.mean(all_accuracy)) if all_accuracy else 0.0,
            knowledge_over_time=knowledge_over_time,
            weakest_over_time=weakest_over_time,
        ))

        logger.info(
            f"  {algo_name}: avg_k={results[-1].avg_knowledge_final:.4f}, "
            f"weakest={results[-1].weakest_knowledge_final:.4f}, "
            f"acc={results[-1].accuracy:.4f}"
        )

    return results


def replay_results_to_dataframe(results: List[ReplayResult]) -> pd.DataFrame:
    """Convert replay results to a DataFrame for CSV export."""
    rows = []
    for r in results:
        row = {
            "algorithm": r.algorithm,
            "dataset": r.dataset_name,
            "num_students": r.num_students,
            "num_interactions": r.num_interactions,
            "avg_knowledge_final": r.avg_knowledge_final,
            "weakest_knowledge_final": r.weakest_knowledge_final,
            "accuracy": r.accuracy,
        }
        # Add trajectory data
        for i, (k, w) in enumerate(zip(r.knowledge_over_time, r.weakest_over_time)):
            row[f"knowledge_t{i}"] = k
            row[f"weakest_t{i}"] = w
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fitted-parameter simulation
# ---------------------------------------------------------------------------

def run_fitted_simulation(
    fitted_params: FittedParams,
    algorithms: List[str],
    num_categories: int = 20,
    num_students: int = 100,
    questions_per_session: int = 2000,
    seed: int = 42,
) -> "MultiAlgorithmResults":
    """
    Run synthetic simulation using parameters fitted from real data.

    This creates an ExperimentConfig with the fitted parameters and runs
    MultiAlgorithmExperiment, providing a bridge between real data and
    our synthetic framework.
    """
    from .simulation import MultiAlgorithmExperiment

    config = ExperimentConfig(
        num_students_per_group=num_students,
        num_categories=num_categories,
        initial_knowledge_mean=fitted_params.initial_knowledge_mean,
        initial_knowledge_std=0.1,
        learning_rate=fitted_params.learning_rate,
        incorrect_penalty=fitted_params.incorrect_penalty,
        base_knowledge=fitted_params.base_knowledge,
        decay_rate=fitted_params.decay_rate,
        questions_per_session=questions_per_session,
        random_seed=seed,
    )

    exp = MultiAlgorithmExperiment(config, algorithms)
    return exp.run(show_progress=True)


# ---------------------------------------------------------------------------
# CLI entry point for real data experiments
# ---------------------------------------------------------------------------

def run_real_data_experiment(
    dataset_name: str,
    data_dir: str,
    algorithms: List[str],
    output_dir: str,
    max_students_fit: int = 200,
    max_students_replay: int = 200,
    max_interactions: int = 500,
    fitted_params_path: Optional[str] = None,
    run_replay: bool = True,
    run_fitted_sim: bool = True,
    seed: int = 42,
) -> None:
    """
    Full real data experiment pipeline.

    1. Load dataset
    2. Fit student model parameters (or load pre-fitted)
    3. Run replay evaluation
    4. Run fitted-parameter simulation
    5. Save all results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if dataset_name == "duolingo":
        dataset = load_duolingo(
            os.path.join(data_dir, "duolingo"),
            max_students=max(max_students_fit, max_students_replay),
        )
    elif dataset_name == "assistments":
        dataset = load_assistments(
            os.path.join(data_dir, "assistments"),
            max_students=max(max_students_fit, max_students_replay),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loaded {dataset.name}: {dataset.num_students} students, "
          f"{dataset.num_skills} skills, {dataset.num_interactions} interactions")

    # Fit or load parameters
    if fitted_params_path and Path(fitted_params_path).exists():
        print(f"Loading pre-fitted params from {fitted_params_path}")
        fitted = FittedParams.load(fitted_params_path)
    else:
        print("Fitting student model parameters...")
        fitted = fit_student_model(dataset, max_students=max_students_fit)
        params_path = os.path.join(output_dir, f"fitted_params_{dataset_name}.json")
        fitted.save(params_path)
        print(f"Saved fitted params to {params_path}")

    print(f"\nFitted parameters:")
    print(f"  α (learning_rate):      {fitted.learning_rate:.4f}")
    print(f"  β (incorrect_penalty):  {fitted.incorrect_penalty:.4f}")
    print(f"  λ (decay_rate):         {fitted.decay_rate:.4f}")
    print(f"  base_knowledge:         {fitted.base_knowledge:.4f}")
    print(f"  initial_knowledge:      {fitted.initial_knowledge_mean:.4f}")
    print(f"  fit_loss (NLL/obs):     {fitted.fit_loss:.4f}")

    # Replay evaluation
    if run_replay:
        print(f"\nRunning replay evaluation on {dataset_name}...")
        replay_results = replay_evaluate(
            dataset, fitted, algorithms,
            max_students=max_students_replay,
            max_interactions_per_student=max_interactions,
        )

        print(f"\nReplay Results ({dataset_name}):")
        print("{:20s} | {:>14s} | {:>14s} | {:>10s}".format(
            "Algorithm", "Avg Knowledge", "Weakest Know.", "Accuracy"))
        print("-" * 65)
        for r in replay_results:
            print("{:20s} | {:14.4f} | {:14.4f} | {:10.4f}".format(
                r.algorithm, r.avg_knowledge_final,
                r.weakest_knowledge_final, r.accuracy))

        replay_df = replay_results_to_dataframe(replay_results)
        replay_path = os.path.join(output_dir, f"replay_results_{dataset_name}.csv")
        replay_df.to_csv(replay_path, index=False)
        print(f"Saved replay results to {replay_path}")

    # Fitted-parameter simulation
    if run_fitted_sim:
        print(f"\nRunning fitted-parameter simulation ({dataset_name} params)...")
        sim_results = run_fitted_simulation(
            fitted, algorithms,
            num_categories=min(dataset.num_skills, 50),
            num_students=100,
            questions_per_session=2000,
            seed=seed,
        )

        sim_path = os.path.join(output_dir, f"fitted_sim_{dataset_name}.csv")
        sim_results.save_to_csv(sim_path)
        print(f"Saved fitted simulation results to {sim_path}")

        # Print final summary
        print(f"\nFitted Simulation Results ({dataset_name} params):")
        print("{:20s} | {:>14s} | {:>14s}".format(
            "Algorithm", "Avg Knowledge", "Weakest Know."))
        print("-" * 55)
        for algo in algorithms:
            final = sim_results.metrics[algo][-1]
            print("{:20s} | {:14.4f} | {:14.4f}".format(
                algo, final.average_knowledge, final.weakest_category_avg))
