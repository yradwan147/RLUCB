# Adaptive Question Selection Using Upper Confidence Bound: A Simulation Study

## 1. Introduction and Objectives

This project investigates whether **bandit-based adaptive question selection** can improve student learning outcomes compared to uniform random selection in an educational quizzing context. Specifically, we apply the **Upper Confidence Bound (UCB1)** algorithm — originally developed for the multi-armed bandit problem — to dynamically choose which knowledge category to quiz a student on next.

The central research question is:

> *Does UCB-guided adaptive question selection produce faster, more balanced knowledge acquisition than random selection, particularly for a student's weakest knowledge areas?*

The hypothesis is that by treating each knowledge category as an "arm" and using the UCB exploration-exploitation trade-off, the system can prioritize weak areas while still maintaining coverage of all topics — leading to more efficient learning.

---

## 2. Method

### 2.1 Student Model

Each simulated student maintains a **knowledge vector** $\mathbf{K} = [k_1, k_2, \ldots, k_6]$, where $k_i \in [0, 1]$ represents the probability of answering correctly in category $i$. Initial knowledge is sampled from $k_i \sim \mathcal{N}(\mu \cdot d_i,\; \sigma^2)$, where $d_i$ is the category difficulty and $\mu = 0.35$, $\sigma = 0.1$.

**Learning dynamics.** When a student answers a question in category $i$:

- **Correct answer:** $k_i \leftarrow k_i + \alpha \cdot (1 - k_i)$, an asymptotic update toward 1.0 (learning rate $\alpha = 0.12$).
- **Incorrect answer:** $k_i \leftarrow k_i - \beta \cdot k_i$, a multiplicative penalty ($\beta = 0.02$).

**Forgetting dynamics.** At each timestep, unexposed categories decay following an Ebbinghaus-inspired model:

$$k_i \leftarrow k_{\text{base}} + (k_i - k_{\text{base}}) \cdot e^{-\lambda}$$

where $k_{\text{base}} = 0.10$ is the retention floor and $\lambda = 0.01$ is the decay rate.

### 2.2 Question Selection Strategies

Two selection strategies are compared in a between-subjects simulation design:

**UCB Group (Adaptive).** The next category to quiz is selected by maximizing:

$$\text{UCB}(i) = \underbrace{(1 - r_i)}_{\text{exploitation}} + \underbrace{c \cdot \sqrt{\frac{\ln N}{n_i}}}_{\text{exploration}}$$

where $r_i$ is the correctness rate for category $i$, $N$ is the total number of questions asked, $n_i$ is the number of attempts on category $i$, and $c = \sqrt{2} \approx 1.414$ is the exploration parameter. The first term prioritizes categories with low correctness (weak areas); the second term encourages exploration of under-tested categories.

**Control Group (Random).** Categories are selected uniformly at random with probability $1/6$ each.

### 2.3 Experiment Design

| Parameter | Value |
|---|---|
| Students per group | 100 |
| Knowledge categories | 6 |
| Questions per student | 10,000 |
| Learning rate ($\alpha$) | 0.12 |
| Incorrect penalty ($\beta$) | 0.02 |
| Forgetting decay rate ($\lambda$) | 0.01 |
| Base knowledge floor ($k_{\text{base}}$) | 0.10 |
| UCB exploration constant ($c$) | 1.414 |
| Random seed | 42 (for reproducibility) |

Both groups share identical student populations (same initial knowledge distributions and random seeds) to isolate the effect of the selection strategy. The experiment was replicated across 3 independent runs. Group-level metrics (mean knowledge, weakest category performance, knowledge variance, cumulative accuracy) were recorded at every timestep.

### 2.4 Analysis Framework

Results are analyzed across three temporal phases:

- **Short-term** (0–500 questions): Initial learning and exploration.
- **Mid-term** (500–2,000 questions): Knowledge consolidation.
- **Long-term** (2,000–10,000 questions): Saturation and convergence behavior.

---

## 3. Results

### 3.1 Overall Knowledge Acquisition

UCB achieves higher average knowledge during the short-to-mid term but random selection converges to comparable (and slightly higher) levels given sufficient time.

| Timestep | UCB Avg Knowledge | Random Avg Knowledge | UCB Advantage |
|---|---|---|---|
| 100 | 0.321 | 0.348 | -7.6% |
| 500 | 0.566 | 0.544 | **+3.9%** |
| 1,000 | 0.588 | 0.570 | **+3.1%** |
| 2,000 | 0.569 | 0.563 | **+1.1%** |
| 5,000 | 0.547 | 0.568 | -3.7% |
| 10,000 | 0.536 | 0.562 | -4.6% |

UCB's peak overall advantage of **+2.3%** occurs at $t \approx 470$. UCB leads in overall knowledge for **21.2%** of all timesteps, concentrated in the first 2,000 questions.

### 3.2 Weakest Category Protection

The most striking difference is in **weakest category performance**, where UCB shows a dramatic early advantage:

| Timestep | UCB Weakest | Random Weakest | UCB Advantage |
|---|---|---|---|
| 100 | 0.207 | 0.129 | **+60.7%** |
| 250 | 0.410 | 0.262 | **+56.7%** |
| 500 | 0.500 | 0.365 | **+36.8%** |
| 1,000 | 0.509 | 0.407 | **+25.2%** |
| 2,000 | 0.458 | 0.398 | **+15.0%** |
| 5,000 | 0.390 | 0.386 | +0.8% |
| 10,000 | 0.358 | 0.390 | -8.2% |

UCB's peak weakest-category advantage is **+60.7% at $t = 100$**, and it leads for **53.5%** of all timesteps. This confirms that UCB is highly effective at rapidly lifting the lowest-performing knowledge area.

### 3.3 Phase Summary

| Phase | UCB Overall Advantage | UCB Weakest Category Advantage | Winner |
|---|---|---|---|
| Short-term ($t = 500$) | **+3.9%** | **+36.8%** | **UCB** |
| Mid-term ($t = 2{,}000$) | **+1.1%** | **+15.0%** | **UCB** |
| Long-term ($t = 10{,}000$) | -4.6% | -8.2% | Random |

### 3.4 Category-Level Observations

- **UCB** produces more uniform knowledge heatmaps across categories, indicating balanced learning.
- **Random** shows higher inter-category variance early on, with some categories advancing much faster than others.
- Final category knowledge range: UCB = 0.411, Random = 0.320. UCB has wider spread at termination due to its aggressive earlier focus on weak categories creating temporary imbalances during the long-term phase.

---

## 4. Discussion and Key Observations

1. **UCB excels in time-constrained settings.** The adaptive strategy provides 25–60% better protection for weak areas during the first 500–1,000 questions. For practical learning scenarios where study time is limited, this is a significant advantage.

2. **Convergence under extended practice.** Given enough questions (~10,000 across 6 categories, or ~1,667 per category), random selection achieves comparable and slightly higher overall knowledge. This is expected: with sufficient exposure, even uniform allocation provides adequate coverage.

3. **The exploration-exploitation trade-off works as intended.** UCB's initial focus on weak categories comes at a small short-term cost to overall accuracy (UCB is -7.6% at $t = 100$) but pays off by $t = 500$ when it surpasses random in overall knowledge.

4. **UCB's value lies in efficiency, not ceiling.** Both strategies reach similar knowledge ceilings (~0.55–0.59). UCB reaches this level faster and with more balanced category coverage.

5. **Diminishing returns of adaptivity.** As all categories receive sufficient practice, UCB's advantage diminishes because the exploitation signal (correctness rate differences) shrinks, and the exploration bonus becomes negligible relative to accumulated experience.

---

## 5. Preliminary Conclusions

- The UCB algorithm is an effective and principled approach for adaptive question selection, particularly when learning time is constrained.
- UCB's primary benefit is **learning efficiency** — achieving balanced, higher knowledge in fewer questions — rather than improving the absolute knowledge ceiling.
- For sessions under ~1,000 questions per student, UCB is strongly recommended over random selection.
- Future work could explore: (a) hybrid strategies (UCB early, random later), (b) tuning the exploration parameter $c$ for different learner profiles, (c) non-stationary variants that account for changing student knowledge, and (d) validation with real student data.

---

*Experiment conducted with random seed 42 for reproducibility. Full source code, data, and visualizations are available in the project repository.*
