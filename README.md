# Adaptive Question Selection via Upper Confidence Bound (UCB)

A reinforcement learning experiment comparing **UCB-based adaptive quizzing** against **random quizzing** for simulated student learning, with an accompanying **NeurIPS-formatted research paper**.

## Key Findings

| Metric | UCB Advantage | When |
|--------|---------------|------|
| Peak Weakest Category | **+60.7%** | Short-term (t=100) |
| Peak Overall Knowledge | **+3.9%** | Mid-term (t=500) |
| Time Ahead (Weakest) | **53.5%** | Majority of experiment |

UCB achieves dramatically faster improvement in weak knowledge areas. Given unlimited time (~10,000 questions), both strategies converge to similar knowledge ceilings.

## Overview

This project simulates two groups of 100 students across 6 knowledge categories and 10,000 questions per session:
- **UCB Group**: Questions selected using the Upper Confidence Bound algorithm (adaptive, focuses on weak areas)
- **Control Group**: Questions selected uniformly at random

The simulation incorporates realistic asymptotic learning dynamics and Ebbinghaus-inspired exponential forgetting.

## Installation

```bash
pip install -r requirements.txt
```

Required packages: `gymnasium`, `numpy`, `matplotlib`, `seaborn`, `pandas`, `tqdm`

## Quick Start

```bash
# Run with defaults and display dashboard
python run_experiment.py

# Run the full experiment (100 students, 10K questions) and save everything
python run_experiment.py --students 100 --questions 10000 --dashboard --csv --output results
```

## CLI Reference

```
python run_experiment.py [OPTIONS]
```

### Population Settings

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--students` | `-s` | 50 | Number of students per group |
| `--categories` | `-c` | 6 | Number of knowledge categories |

### Learning Dynamics

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--learning-rate` | `-lr` | 0.12 | Knowledge increase on correct answer (alpha) |
| `--penalty` | | 0.02 | Knowledge decrease on incorrect answer (beta) |

**Learning formula:**
- Correct: `k_new = k + alpha * (1 - k)` (asymptotic toward 1.0)
- Incorrect: `k_new = k - beta * k` (multiplicative penalty)

### Forgetting Dynamics

| Argument | Default | Description |
|----------|---------|-------------|
| `--decay-rate` | 0.01 | Exponential decay rate per timestep |
| `--base-knowledge` | 0.10 | Minimum retained knowledge (retention floor) |

**Forgetting formula (Ebbinghaus-inspired):**
```
k_new = base + (k - base) * exp(-decay_rate)
```

### Experiment Settings

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--questions` | `-q` | 200 | Number of questions per student session |
| `--exploration` | | 1.414 | UCB exploration parameter (c), typically sqrt(2) |
| `--seed` | | 42 | Random seed for reproducibility |
| `--runs` | | 1 | Number of experiment runs (for statistical analysis) |

### Output Settings

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--output` | `-o` | `results` | Output directory for saved files |
| `--dashboard` | | False | Generate comprehensive 8-panel dashboard |
| `--individual-plots` | | False | Generate separate plot files |
| `--csv` | | False | Export results to CSV file |
| `--no-show` | | False | Don't display plots (only save to files) |
| `--quiet` | `-Q` | False | Suppress progress bars and summary output |

## Usage Examples

```bash
# Basic run with interactive dashboard
python run_experiment.py

# Large-scale experiment
python run_experiment.py --students 100 --questions 10000

# Save all outputs
python run_experiment.py -s 50 -q 300 --dashboard --individual-plots --csv --output my_experiment

# Batch processing (no display)
python run_experiment.py -s 100 -q 500 --dashboard --csv --no-show --output batch_results

# Multiple runs for statistical analysis
python run_experiment.py --runs 10 --csv --output statistical_analysis

# Quick test
python run_experiment.py -s 10 -q 50 --no-show
```

## Programmatic Usage

```python
from experiment.config import ExperimentConfig
from experiment.simulation import Experiment
from experiment.visualization import create_dashboard, print_summary_statistics

config = ExperimentConfig(
    num_students_per_group=100,
    num_categories=6,
    questions_per_session=10000,
    learning_rate=0.12,
    decay_rate=0.01,
    random_seed=42,
)

experiment = Experiment(config)
results = experiment.run()

print(print_summary_statistics(results))
create_dashboard(results, save_path="dashboard.png")
results.save_to_csv("results.csv")
```

## Project Structure

```
RLUCB/
├── experiment/                  # Simulation framework
│   ├── config.py                #   ExperimentConfig dataclass
│   ├── student.py               #   Student model (learning + forgetting)
│   ├── selectors.py             #   RandomSelector, UCBSelectorAdapter
│   ├── environment.py           #   Gymnasium RL environment
│   ├── simulation.py            #   Experiment runner + metrics
│   └── visualization.py         #   Plotting and dashboard functions
├── UCBQuizMaster/               # Core UCB algorithm
│   └── ucb_algorithm.py         #   UCBSelector class
├── paper/                       # NeurIPS-formatted research paper
│   ├── main.tex                 #   Main LaTeX document
│   ├── main.pdf                 #   Compiled paper (13 pages)
│   ├── sec/                     #   Paper sections (abstract through conclusion)
│   ├── tables/                  #   Standalone table files
│   ├── figs/                    #   Figures from experiments
│   └── refs.bib                 #   Bibliography (22 citations)
├── report_data/                 # Main experiment results
│   ├── experiment_results.csv   #   Full timestep-by-timestep data
│   ├── EXPERIMENT_REPORT.md     #   Detailed analysis report
│   └── fig*.png                 #   Result visualizations
├── run_experiment.py            # CLI entry point
├── research_summary.md          # Research summary with methodology
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Output Files

When using `--output results`:

| File | Flag | Description |
|------|------|-------------|
| `dashboard.png` | `--dashboard` | Comprehensive 8-panel visualization |
| `experiment_results.csv` | `--csv` | Full timestep-by-timestep data |
| `knowledge_comparison.png` | `--individual-plots` | Average knowledge over time |
| `category_heatmaps.png` | `--individual-plots` | Per-category knowledge evolution |
| `exposure_distribution.png` | `--individual-plots` | Category quiz frequency |
| `weakest_category.png` | `--individual-plots` | Weakest area improvement |
| `knowledge_variance.png` | `--individual-plots` | Knowledge uniformity over time |
| `final_distribution.png` | `--individual-plots` | Final knowledge violin plots |
| `accuracy.png` | `--individual-plots` | Cumulative answer accuracy |

## Research Paper

The `paper/` directory contains a complete NeurIPS 2025-formatted research paper (`paper/main.pdf`) covering:

- **Problem formulation**: Adaptive question selection as a multi-armed bandit problem
- **Methodology**: Student model with asymptotic learning and Ebbinghaus forgetting, UCB1 with inverted reward signal
- **Experiments**: 100 students/group, 6 categories, 10K questions, analyzed across three temporal phases
- **Key results**: UCB provides up to 60.7% better weakest-category performance; both converge long-term

To recompile the paper:
```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## License

MIT
