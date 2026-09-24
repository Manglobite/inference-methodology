# Скрипты исследования

> Шаблон. Скелет **не содержит скриптов**: этот каталог поставляется пустым
> (только этим README). Нужные скрипты копируются в `<кейс>/scripts/` из
> канонических переносимых версий методологии: `../../scripts/<категория>/`
> (путь относительно этого каталога при копировании скелета).
>
> Здесь описано, какие скрипты бывают, из какой категории их брать и зачем.
> При копировании скелета в каталог кейса пути пересчитываются от нового
> расположения (см. `METHODOLOGY.md`, раздел про адаптацию).

## Категории скриптов методологии

| Категория | Что лежит | Копировать в кейс |
| --- | --- | --- |
| `scripts/runner/` | Раннер прогонов (`run_cache_sessions.py`) и серверные обёртки режимов (`run-serve-matrix.sh`, `run-bench-matrix.sh`). | Да, при постановке измерений. |
| `scripts/telemetry/` | Сбор телеметрии хоста/GPU (`host_telemetry.py`, схема версии 1). | Да, если нужна покадровая телеметрия. |
| `scripts/report/` | Агрегация сырых прогонов (`generate_report.py`) и построение графиков (`plot_context_curves.py`). | Да, для агрегатов и графиков. |
| `scripts/publish/` | Барьер публикации (`check-public.sh`) и санитайз артефактов (`sanitize-results.sh`). | Обязательно перед публикацией. |

Скрипты, специфичные для узла (например, снятие/возврат idle-политики control
plane `hub-idle.sh`), в канонический набор методологии не входят: их кладут в
`<кейс>/scripts/` отдельно, если такой control plane есть. Обёртки вызывают
`hub-idle.sh` только при его наличии.

## Как переносить

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

Нужны не все скрипты — копируйте только используемые. Барьер публикации
(`check-public.sh`) и санитайз (`sanitize-results.sh`) нужны всегда.

## Соглашения

- Все пути внутри скриптов — относительные к каталогу кейса или к корню
  репозитория (`--case-dir`/`--repo-root`); абсолютных домашних путей нет.
- Профили содержат плейсхолдер `<repo>`; раннер подставляет корень, заданный
  `--repo-root` (по умолчанию — каталог кейса).
- Раннер запускает и останавливает **собственный** сервер, поэтому прогоны
  выполняются по одному.
- Сырые прогоны не перезаписываются; неудачные сохраняются со `status = failed`.
- Переносимые скрипты требуют только стандартной библиотеки Python (без
  `numpy`) или `bash`; числа GPU/ядер определяются динамически.

## Пример вызова

```bash
# из каталога кейса
python3 scripts/host_telemetry.py results/<run_id>/telemetry.csv 0.5 --proc-name llama-server
python3 scripts/run_cache_sessions.py --profile profiles/<name>.json --mode ladder --case-dir .
python3 scripts/generate_report.py --results-dir results --docs-dir docs
python3 scripts/plot_context_curves.py --results-dir results --out-dir docs/figures --lang ru
bash scripts/check-public.sh
bash scripts/sanitize-results.sh
```

Подробный индекс переносимых скриптов с назначением, вызовом и требованиями —
в README методологии (`../../scripts/README.md`).
