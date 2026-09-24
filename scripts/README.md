# Скрипты методологии: индекс переносимых версий

Канонические **переносимые** версии скриптов методологии. Они не привязаны к
конкретному железу и кейсу: каталог кейса, корень репозитория, порт и устройства
задаются аргументами или переменными окружения, число GPU/ядер определяется
динамически.

**Два разных дерева.** В самой методологии скрипты лежат по категориям:
`scripts/runner/…`, `scripts/telemetry/…`, `scripts/report/…`, `scripts/publish/…`.
При переносе в кейс они копируются **в плоский каталог** `<case>/scripts/`
(см. «Перенос в кейс»). Поэтому в таблицах ниже и пути в колонке «Скрипт», и
команды в колонке «Вызов» приведены **как в каноническом дереве методологии**
(`scripts/runner/…`, `scripts/report/…` и т. д.); после flatten-копирования в
кейсе те же команды вызываются по плоским путям (`scripts/…`).

Требования: Python-скрипты используют **только стандартную библиотеку**; обёртки
и барьер — **bash**. Скрипты, специфичные для узла (например, управление
idle-политикой control plane), в канонический набор не входят.

## runner/ — прогоны

| Скрипт | Назначение | Вызов | Требования |
| --- | --- | --- | --- |
| `run_cache_sessions.py` | Раннер измерений: `smoke` (проверка запуска), `ladder` (лесенка 10/30/60/80 % контекста), `ab-sequence` (чередование двух сессий по строке порядка). Поднимает свой `llama-server`, шлёт запросы и пишет `results/<run_id>/` (`command.json`, `profile.json`, `server.log`, `telemetry.csv`, `result.json`). Телеметрия — самодостаточная, встроенная. | `python3 scripts/runner/run_cache_sessions.py --profile profiles/<name>.json --mode ladder --case-dir <кейс> --repo-root <корень>` | Python stdlib; опционально `nvidia-smi` (при его отсутствии не падает) |
| `run-serve-matrix.sh` | Обёртка запуска профиля: `smoke` (прогон с проверкой) или `up` (detached-сервер). `CASE_DIR` — родитель каталога скрипта, `ROOT_DIR=${REPO_ROOT:-$CASE_DIR}`; `hub-idle.sh` вызывается только при наличии. | `bash scripts/runner/run-serve-matrix.sh <profile> [smoke\|up]` | bash; профиль в `<case>/profiles/<profile>.json` |
| `run-bench-matrix.sh` | Обёртка измерений: `ladder`, `ab-abab`, `ab-abbabaa`, а также `report`, `plot`, `all`, `check`, `sanitize`, `sanitize-fix`. `report`/`plot`/`all` вызывают `generate_report.py`/`plot_context_curves.py` из `<case>/scripts/`, передавая абсолютные `--results-dir`/`--docs-dir`/`--out-dir` и (при наличии) `--config "$CASE_DIR/study.json"`, поэтому режим не зависит от cwd/`REPO_ROOT`; без `study.json` — автоопределение. `check` — `check-public.sh`, `sanitize`/`sanitize-fix` — `sanitize-results.sh`. | `bash scripts/runner/run-bench-matrix.sh <profile> <ladder\|ab-abab\|ab-abbabaa>` или `bash scripts/runner/run-bench-matrix.sh report\|plot\|all\|check\|sanitize\|sanitize-fix` | bash |

Ключевые аргументы раннера: `--profile` (обязателен), `--mode`
(`smoke|ladder|ab-sequence`), `--case-dir` (по умолчанию `cwd`), `--repo-root`
(по умолчанию `--case-dir`; в нём заменяется плейсхолдер `<repo>` и запускается
сервер), `--results-dir` (по умолчанию `<case-dir>/results`), `--seed` (42),
`--order` (`ABAB`), `--port`, `--clock-reset-devices`. Сброс частот GPU
выполняется, только если устройства заданы аргументом `--clock-reset-devices`
или ключом профиля `clock_reset_devices`; иначе частоты не трогаются, а пропуск
фиксируется в `result.json`.

## telemetry/ — телеметрия

| Скрипт | Назначение | Вызов | Требования |
| --- | --- | --- | --- |
| `host_telemetry.py` | Сбор телеметрии по нормативной схеме `METHODOLOGY.md` §10 (версия схемы 1): `schema_version`, `timestamp_utc`, CPU/RAM/swap, per-core `cpuN_pct`, по каждому GPU `gpuN_temp_c`/`gpuN_util_pct`/`gpuN_mem_used_mib`/`gpuN_power_w`/`gpuN_sm_clock_mhz`/`gpuN_pstate`, RSS/PSS/swap сервера и `llama_pids`. Интервал по умолчанию 0.5 с; существующий файл не перезаписывается без `--force`. | `python3 scripts/telemetry/host_telemetry.py <csv> [interval] [--pid N] [--proc-name llama-server]` | Python stdlib; опционально `nvidia-smi` |

## report/ — агрегация и графики

| Скрипт | Назначение | Вызов | Требования |
| --- | --- | --- | --- |
| `generate_report.py` | Агрегация сырых прогонов (`<results-dir>/*/result.json`) в `<docs-dir>/results-tables.md` и `<docs-dir>/results.json`. Read-only для прогонов; каноническая агрегация — по `ladder`/`ok`/`completion_is_fixed` (§11.1 METHODOLOGY.md), норматив `n≥3` — по каждой ступени `target_pct`. Конфигурация — `study.json`; автоопределение только при отсутствии `./study.json` и не заданном явно `--config`; существующий, но невалидный конфиг — ошибка (`exit 2`). | `python3 scripts/report/generate_report.py [--results-dir results] [--docs-dir docs] [--config study.json] [--case-dir <кейс>] [--repo-root <корень>]` | Python stdlib |
| `plot_context_curves.py` | Построение графиков в `docs/figures/*.svg` (`prefill-vs-context`, `decode-vs-context`, `ab-cache-hit`), ru + en, без внешних зависимостей (hand-written SVG). Скрабинг абсолютных путей через `--case-dir`/`--repo-root` (как у агрегатора). При отсутствии конфига подписи профилей не раскрываются: в легенду вместо имени идёт нейтральный плейсхолдер (`профиль N`/`profile N`), а предупреждение в логе содержит только счётчик без имён. | `python3 scripts/report/plot_context_curves.py [--results-dir results] [--out-dir docs/figures] [--config study.json] [--lang ru\|en] [--case-dir <кейс>] [--repo-root <корень>]` | Python stdlib |

**Область графиков.** `plot_context_curves.py` автоматически строит **только
три фигуры**: `prefill-vs-context`, `decode-vs-context`, `ab-cache-hit`. Графики
по осям раскладки, `--ubatch-size` и глубины спекуляции (METHODOLOGY.md §11.2)
автор строит **отдельно** — вручную или своим скриптом; `plot_context_curves.py`
их не генерирует. При наличии соответствующей измеренной серии такой график
обязателен по §11.2; если серия не измерялась, отсутствие графика объясняется в
`README.md` кейса.

## publish/ — публикация

| Скрипт | Назначение | Вызов | Требования |
| --- | --- | --- | --- |
| `check-public.sh` | Барьер публикации: рекурсивный обход дерева кейса с исключением служебных каталогов (`.git/`, `build/`, `dist/`, `__pycache__/`, `node_modules/`, `*.pyc`) и `grep` по обобщённым паттернам приватности. Baseline: `/home/<user>/`, `/mnt/`, `/root/`, `/Users/<user>/`, RFC1918 (10/8, 172.16/12, 192.168/16), секреты (точный паттерн `sk-[A-Za-z0-9_-]{13,}`, `Authorization: Bearer`, `api_key`/`secret`/`token`). Имена хостов и внутренние идентификаторы baseline не детектирует — добавляются через `PUBLIC_DENY_FILE`. Fail-closed: `exit 2` при находке/ошибке, `exit 0` только при чистоте. Ложные срабатывания снимаются `.public-allow`. | `bash scripts/publish/check-public.sh` | bash (`find`, `grep`) |
| `sanitize-results.sh` | Санитайз артефактов: заменяет приватные пути, точки монтирования, RFC1918-адреса, секреты и пользовательские маркеры на переносимые плейсхолдеры. Без `--in-place` — только отчёт (`exit != 0` при находках); с `--in-place` — перезапись. Узел-специфичные данные инжектируются через `SANITIZE_USER`/`SANITIZE_REPO_ROOT`/`--patterns`. | `bash scripts/publish/sanitize-results.sh [--in-place] [--patterns FILE] [path ...]` | bash (`find`, `grep`, `sed`) |

## Перенос в кейс

При переносе категориальное дерево методологии **схлопывается в плоское**
`<case>/scripts/`: все выбранные скрипты копируются в один каталог независимо от
исходной категории. Ниже — пример flatten-копирования из каталога кейса, когда
методология лежит в `../../scripts/` (пути справа — целевые, плоские):

```bash
# из каталога кейса
mkdir -p scripts
cp ../../scripts/runner/run_cache_sessions.py scripts/
cp ../../scripts/runner/run-serve-matrix.sh scripts/
cp ../../scripts/runner/run-bench-matrix.sh scripts/
cp ../../scripts/telemetry/host_telemetry.py scripts/
cp ../../scripts/report/generate_report.py scripts/
cp ../../scripts/report/plot_context_curves.py scripts/
cp ../../scripts/publish/check-public.sh scripts/
cp ../../scripts/publish/sanitize-results.sh scripts/
chmod +x scripts/*.sh
```

После копирования команды в кейсе вызываются от его корня по плоским путям:
`python3 scripts/run_cache_sessions.py …`, `python3 scripts/generate_report.py …`,
`bash scripts/check-public.sh` (см. `templates/research-case/scripts/README.md`).

Барьер и санитайз нужны всегда; остальные копируются по потребности. Обёртки
рассчитаны на то, что `run_cache_sessions.py` лежит рядом с ними (плоская
раскладка).
