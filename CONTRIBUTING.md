# Как работаем с репозиторием

## Ветки

- `main` — то, что сдаём. Обновляется только PR из `develop` перед сабмитом.
- `develop` — общая рабочая ветка. Прямые пуши запрещены, только PR.
- Рабочие ветки — от `develop`, имя `<тип>/<кратко>`:
  `feat/data-solver-filters`, `fix/llm-usage-parsing`, `docs/readme-cost-model`.

Цикл: ветка от `develop` → коммиты → PR в `develop` → ревью от одного из команды → squash merge → ветка удаляется.

```bash
git switch develop && git pull
git switch -c feat/data-solver-filters
# ... работа ...
git push -u origin feat/data-solver-filters
gh pr create --base develop --fill
```

## Коммиты: Conventional Commits

Формат: `<тип>(<область>): <что сделано>` — в повелительном наклонении, по-английски, без точки в конце.

Типы:

| тип | когда |
|---|---|
| `feat` | новая возможность |
| `fix` | исправление бага |
| `refactor` | переделка без изменения поведения |
| `perf` | ускорение / экономия токенов без изменения результата |
| `test` | тесты |
| `docs` | README, комментарии, презентация |
| `build` | Dockerfile, зависимости |
| `ci` | GitHub Actions |
| `chore` | прочее (gitignore, форматирование) |

Области: `agent`, `llm`, `coding`, `data`, `math`, `search`, `cost-model`, `docker`, `readme`.

Примеры:

```
feat(data): support revenue metric (units*price)
fix(llm): read usage from proxy response headers
perf(coding): skip LLM when mutation fix passes tests
build(docker): pin python 3.11-slim base image
docs(readme): describe token cost estimator
```

Несовместимое изменение — `!` после типа: `feat(agent)!: change solver interface`.

Так как мерджим через squash, в `develop` попадает **заголовок PR** — он тоже должен быть в этом формате.

## Нельзя коммитить

`.idea/`, `__MACOSX/`, `.DS_Store`, `.env` с ключами, логи и траектории (`trajectories*.jsonl`). Всё это уже в `.gitignore`.
