# Hybrid Learning Agent

## Цель проекта

Создать интеллектуального агента, который сочетает:

- Reinforcement Learning
- автоматически извлечённые правила
- память опыта
- логический вывод

## Идея

Большинство RL-агентов учатся только на наградах.

В этом проекте агент дополнительно извлекает знания из своего опыта и превращает их в правила.

Полученные правила используются для ускорения обучения и принятия решений.

## Архитектура

Environment

↓

State Parser

↓

Predicate Generator

↓

Rule Generator

↓

Knowledge Base

↓

Planner

↓

Hybrid Agent

↓

Experience Memory

↓

Evaluation