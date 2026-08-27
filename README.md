# Hybrid Learning Agent New

Универсальная экспериментальная платформа для гибридных интеллектуальных агентов.

Проект объединяет:

- reinforcement learning;
- нейросетевой поиск закономерностей;
- генерацию гипотез;
- статистическую проверку правил;
- символическую базу знаний;
- перенос правил между средами.

## Первая среда

MiniHack / NetHack Learning Environment.

## Запуск проверки

```powershell
docker build -f Dockerfile.minihack -t hybrid-minihack .
docker run --rm hybrid-minihack