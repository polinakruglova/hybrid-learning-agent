# Hybrid Learning Agent

Experimental AI system for discovering transformation rules and causal dependencies directly from interaction with an environment.

The project explores a hybrid approach where an agent does not rely only on a neural network or a predefined symbolic world model. Instead, it observes state transitions, discovers recurring transformations, identifies conditions under which they occur, and builds candidate causal relationships between learned rules.

The current experimental environment is MiniHack.

---

## Core Idea

The agent observes transitions of the form:

```text
State(t)
   ↓
Action
   ↓
State(t+1)
```

Instead of memorizing complete states, the system detects what changed:

```text
Action
   ↓
Changes
```

For example:

```text
action 1
→ x += 1
→ time += 1
```

Repeated observations are converted into transformation rules.

The system then searches for conditions explaining why the same action can produce different effects.

This allows the model to progress from:

```text
action → effect
```

to:

```text
condition + action → effect
```

and eventually toward:

```text
Rule A
   ↓
changes feature X
   ↓
creates a condition for Rule B
   ↓
Rule B
   ↓
...
```

---

## Current Architecture

```text
MiniHack
   │
   ▼
Observation
   │
   ▼
MiniHackStateParser
   │
   ▼
WorldState
   │
   ▼
ChangeDetector
   │
   ▼
TransitionMemory
   │
   ▼
TransformationTracker
   │
   ▼
Base Transformation Rules
   │
   ▼
EffectGrouper
   │
   ▼
ConditionDiscovery
   │
   ▼
Conditional Transformation Rules
   │
   ▼
CausalChainFinder
   │
   ├── CausalLink
   │
   └── Causal Chains
   │
   ▼
CausalAgent
   │
   ▼
Environment
```

---

## State Representation

Raw environment observations are converted into a universal `WorldState`.

Example:

```python
{
    "x": 38,
    "y": 11,
    "time": 4,
    "health": 12,
    "message": ""
}
```

This separates environment-specific observation formats from the learning system.

The long-term goal is to reuse the same learning architecture with different environments and simulators.

---

## Change Detection

`ChangeDetector` compares two consecutive states.

Example:

```text
Before:
x = 38
time = 4

After:
x = 39
time = 5
```

Detected transformation:

```python
{
    "x": ("delta", 1),
    "time": ("delta", 1)
}
```

This gives the system an explicit representation of how the world changed after an action.

---

## Transition Memory

Each interaction is stored approximately as:

```text
Transition
├── state_before
├── action
├── state_after
└── changes
```

The transition history becomes the empirical memory used for rule discovery.

---

## Transformation Rules

Repeated transformations are converted into rules.

Example:

```text
action 1
→ x += 1
```

A rule can also contain conditions.

Example discovered during the MiniHack experiment:

```text
IF x == 40
AND action == 1

THEN

message = "It's solid stone."
```

Conceptually:

```text
conditions + action → effects
```

The rules also store information such as observation count and confidence.

---

## Effect Grouping

The same action does not always produce the same result.

For example, a movement action may:

```text
move successfully
```

or:

```text
hit solid stone
```

or:

```text
produce no state change
```

`EffectGrouper` groups observed transitions according to their resulting effects.

This allows the system to search for the conditions responsible for different outcomes.

---

## Condition Discovery

`ConditionDiscovery` compares states belonging to different effect groups.

For example, the system may observe:

```text
action 3
→ x -= 1
```

in most states, but:

```text
x == 36
+
action 3
→ "It's solid stone."
```

in another group.

A conditional `TransformationRule` can then be created.

The current implementation discovers statistical condition candidates. Some candidates may represent real environmental structure, while others may only be correlations.

Distinguishing correlation from stronger causal evidence is one of the next research problems.

---

## Causal Links

Rules are currently connected when the effect of one rule modifies a feature that appears in the conditions of another rule.

Example:

```text
Rule A
action 1
→ x += 1

        │
        │ changes x
        ▼

Rule B
IF x == 40
AND action 1
→ blocked
```

This produces a candidate causal dependency:

```text
Rule A
   ↓ x
Rule B
```

At the current stage these links should be interpreted as **candidate causal relationships**, not proven causal relationships.

---

## Causal Agent

`CausalAgent` receives:

```text
TransformationRules
+
CausalLinks
```

and uses currently applicable rules when selecting actions.

The agent also keeps exploration enabled through epsilon-based random action selection.

Current experiment:

```text
epsilon = 0.15
```

This allows the system to continue exploring instead of relying exclusively on already discovered rules.

---

# MiniHack Experiment

The current causal-learning pipeline was tested in:

```text
MiniHack-Room-5x5-v0
```

## Rule Discovery

The experiment discovered:

```text
Base transformation rules:        20
Conditional transformation rules: 24
-------------------------------------
Total transformation rules:       44
```

The causal discovery stage produced:

```text
Candidate causal links:  144
Causal chains:           144
```

---

## Agent Evaluation

The learned rules and causal links were then passed to `CausalAgent`.

Evaluation:

```text
Episodes:                 20
Completed episodes:       20 / 20

Average reward:           0.959
Best reward:              0.980
Worst reward:             0.920

Average episode length:   14.10 steps
```

All 20 evaluation episodes terminated successfully.

These results describe the current experiment only and are not intended as a general benchmark against other learning algorithms.

---

# Current Research Direction

The current implementation can discover:

```text
action → effect
```

and:

```text
condition + action → effect
```

It can also construct candidate connections between rules:

```text
Rule A
   ↓ feature
Rule B
```

The next stage is learning causal relationships from actual multi-step trajectories.

Instead of considering rules only pairwise, the system will analyze sequences:

```text
State 0
   ↓ action A
State 1
   ↓ action B
State 2
   ↓ action C
State 3
```

The objective is to determine whether an effect generated by one rule creates the conditions required for a later rule.

Target representation:

```text
Rule A
   ↓
Effect X
   ↓
Condition B becomes true
   ↓
Rule B
   ↓
Effect Y
   ↓
Condition C becomes true
   ↓
Rule C
```

This should allow the system to move from correlation-based rule linking toward learned multi-step causal models.

---

## Long-Term Goal

The project is intended as an environment-independent experimental architecture.

Potential environments include:

- MiniHack / NetHack
- Doom
- MuJoCo
- Isaac Lab
- custom simulation environments

The long-term objective is a hybrid agent capable of learning explicit transformation rules, causal dependencies and reusable models of environment dynamics directly from experience.