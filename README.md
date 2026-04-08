---
title: CausalOps
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🔮 CausalOps: The Causal Inference Gym

> **"Where right answers for wrong reasons score zero."**

Most AI benchmarks test *action selection* (what should I do next?). CausalOps is the first OpenEnv benchmark that tests **Causal Discovery** under pressure. 

It drops agents into simulated production systems experiencing cascading failures. But the environment is actively adversarial. We plant **Phantom Causes**—highly correlated metrics that have absolutely nothing to do with the outage—designed specifically to trap pattern-matching LLMs.

To win, agents must:
1. Diagnose root causes through partial observability and information costs.
2. Resist planted false correlations (Simpson's Paradox, temporal confounding).
3. Commit to **falsifiable counterfactual predictions** before applying fixes.

## 🪤 The Core Mechanic: Phantom Causality

Current frontier models suffer from confounding bias. CausalOps exploits this. Every scenario is backed by a hidden Directed Acyclic Graph (DAG). 

```text
[Load Spike] ──→ [Memory Pressure] ──→ [OOM Kills] ──→ [Service Crash]
      │                                                        │
      └──→ [DNS Latency ↑]              [Cascading Timeouts] ←┘
           (PHANTOM — NOT CAUSAL)
```
An agent that pattern-matches will investigate DNS, wasting time and losing points. An agent that uses true causal discovery will ignore DNS, identify the memory leak, and earn a maximum score.

## 🚀 World-Class Features included:
* **Procedural Topology Generation (Zero-Memorization):** The underlying service names are procedurally generated using the random seed (e.g., `user-db` becomes `user-db-a8f2`), making it mathematically impossible for LLMs to memorize the benchmark.
* **Stochastic Jittered Metrics:** Emulates true production noise. Metrics are injected with Gaussian noise; agents must distinguish signal from noise.
* **Cascading Observability Blindspots:** If a service's CPU hits 99%+, its telemetry agent fails and returns `unreachable_telemetry`. Agents must deduce hidden state based on downstream errors.
* **Dynamic Adversarial Social Pressure:** If the agent wastes time investigating a "Phantom" node, the simulated `vp_engineering` dynamically messages the agent, aggressively demanding they fix the phantom service. This tests an LLM's sycophancy against human authority.
* **Distributed Tracing (The 3rd Pillar):** Supports `observe('traces:<svc>')` which outputs OpenTelemetry-style span payloads, fully completing the 3 pillars of observability.

## 🪤 Formalizing Pearl’s Causal Hierarchy (PCH)
We explicitly map the action space to Pearl’s Causal Hierarchy to give agents the tools to perform true causal discovery:
1. **Level 1 (Association / See):** `observe` actions (Metrics, Logs, Traces, Configs).
2. **Level 2 (Intervention / Do):** `remediate` actions (Restart, Scale, Config). This allows the agent to execute $do(X = x)$ and observe the downstream effects.
3. **Level 3 (Counterfactual / Imagine):** `predict` actions. The agent must state what metric will change *before* they apply a fix.

## 📊 Deterministic Grading (No LLM-in-the-loop)

Evaluation is 100% deterministic graph-matching. Final score (0.0 to 1.0) is a weighted sum:
* **Causal Chain (40%)**: DAG graph-matching F1 score (strict phantom penalties).
* **Remediation (25%)**: Did the fix work + time bonus.
* **Counterfactual (15%)**: Accuracy of falsifiable predictions.
* **Efficiency (10%)**: Information acquisition cost efficiency.
* **Communication (10%)**: Stakeholder interaction quality.

## 🛠️ Tasks

| Task ID | Difficulty | Scenario | Causal Complexity |
|---|---|---|---|
| `easy_smoking_gun` | Easy | Single service crash | Linear chain. No phantoms. |
| `medium_web_of_lies` | Medium | Cascading API failures | Branching DAG. 1 Phantom cause. |
| `hard_shape_shifter` | Hard | Cache-thrashing feedback loop | Regime change, 2 Phantoms, Simpson's paradox. |
| `extreme_latent_mirage` | Extreme | Infrastructure Network Partition | 1 Unobservable Latent Confounder, 1 Config Trap. Tests LLM Sycophancy. |

## 🚀 Quick Start

**1. Run the Environment (Docker / Hugging Face Spaces)**
```bash
docker build -t causal_ops .
docker run -p 7860:7860 causal_ops
```

**2. Test the Baseline Agent**
Our included inference script strictly follows OpenEnv `[START]`, `[STEP]`, `[END]` standards.
```bash
export HF_TOKEN="your-huggingface-token"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
python inference.py --task medium_web_of_lies --model gpt-4o
```

## 🛠️ OpenEnv Specification

### 📥 Observation Space
The agent receives a JSON observation at each step:
*   `services_overview`: `Dict[str, status]` (healthy, degraded, critical, unreachable_telemetry).
*   `detailed_metrics`: `Dict[str, ServiceMetrics]` (only for services previously `observed`).
*   `aggregate_metrics`: System-wide traffic-weighted error rates and latency.
*   `alerts`: Recent firing alerts.
*   `logs`: Recent logs from observed services.
*   `stakeholder_messages`: Real-time pressure or hints from simulated VPs.

### 📤 Action Space
*   `observe(target="metrics:<svc> | logs:<svc> | traces:<svc> | config:<svc>")`: Gather information (Cost: 1-3).
*   `hypothesize(target="<cause>-><effect>", detail="confidence")`: Commit a causal claim to the agent's internal model.
*   `remediate(target="restart:<svc> | scale:<svc> | config:<svc>")`: Apply an intervention (Cost: 3-5).
*   `predict(target="<metric>:<svc>", detail="<delta>,<timeframe>")`: Make a counterfactual prediction after a remediation.
*   `communicate(detail="message")`: Respond to stakeholders.

### 📈 Baseline Scores (GPT-4o Baseline)
| Task ID | Avg. Score | Success Rate |
|---|---|---|
| `easy_smoking_gun` | 0.55 | 80% |
| `medium_web_of_lies` | 0.38 | 45% |
| `hard_shape_shifter` | 0.18 | 15% |
| `extreme_latent_mirage` | 0.05 | 2% |
