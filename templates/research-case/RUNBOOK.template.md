# RUNBOOK: воспроизведение

> Шаблон. Заполните фактические команды, провенанс и диагностику своего кейса.
> Скрипты кейса — копии переносимых версий методологии
> (`methodology/scripts/<категория>/` → `<case>/scripts/`); обёртки запускаются
> **из каталога кейса** (`scripts/...`) и сами определяют корень
> inference-репозитория, разрешая профили относительно него.

Полный отчёт — в [README.md](README.md); конфигурация стенда — в
[HARDWARE.md](HARDWARE.md); лицензии — в [LICENSE-NOTICE.md](LICENSE-NOTICE.md).

## 1. Окружение и предполётные проверки

- Свободные GPU: `nvidia-smi` (один тяжёлый процесс за раз).
- Место на диске: `df -h` и `df -h` для тома с моделями.
- Прод-бэкенд узла остановить (если есть control plane), чтобы освободить GPU.
- Снять политику auto-idle узла и разбудить GPU (если control plane её имеет).
- Права: скрипты могут быть без бита исполнения — вызывайте их через
  `bash scripts/<name>.sh`, а не `./scripts/...`.
- Конфиг `study.json` в корне кейса (или его осознанное отсутствие — тогда
  агрегатор автоопределяет профили из прогонов).
- Git: каталог кейса публикуется файловым барьером (раздел 6), git-операции
  не требуются.

## 2. Провенанс: веса

Веса **не включаются** в репозиторий; воспроизводятся только источник и
sha256.

```bash
# скачивание (замените URL/commit/file на свои)
curl -L -o data/models/<file>.gguf \
  <url>/resolve/<commit>/<file>.gguf

# проверка размера и контрольной суммы
sha256sum data/models/<file>.gguf
# ожидается <sha256> (<bytes> Б)
```

## 3. Провенанс: рантаймы

ЗАПОЛНИТЬ: движок, commit, флаги сборки, путь к бинарю, манифест. Если
пересборка не требуется — так и напишите.

| Устройство | Бинарь |
| --- | --- |
| ЗАПОЛНИТЬ | `<repo>/tools/llama-cpp/<RUNTIME>/bin/llama-server` |

Происхождение (commit, флаги сборки, sha256 файлов) — в
`tools/llama-cpp/<RUNTIME>/manifest.json`.

## 4. Прогоны через обёртки

ЗАПОЛНИТЬ: список профилей, порты, режимы.

### 4.1. Smoke-проверка

```bash
bash scripts/run-serve-matrix.sh <profile> smoke
```

Запускает сервер, проверяет `/props`, шлёт короткий запрос и останавливает
сервер; пишет `results/<id>/result.json`.

### 4.2. Основные прогоны (ladder / A-B)

```bash
bash scripts/run-bench-matrix.sh <profile> ladder
bash scripts/run-bench-matrix.sh <profile> ab-abab
bash scripts/run-bench-matrix.sh <profile> ab-abbabaa
```

ЗАПОЛНИТЬ: какие ступени, какие порядки A/B, для каких конфигураций.

## 5. Агрегация

Скрипты агрегации — копии `methodology/scripts/report/`. Конфиг `study.json`
(из `study.example.json`) кладётся в корень кейса; при его отсутствии профили,
метки, порядки A/B и ступени автоопределяются из сырых прогонов. Агрегаторы
read-only по отношению к прогонам:

```bash
python3 scripts/generate_report.py --results-dir results --docs-dir docs --config study.json
python3 scripts/plot_context_curves.py --results-dir results --out-dir docs/figures --lang ru --config study.json
python3 scripts/plot_context_curves.py --results-dir results --out-dir docs/figures --lang en --config study.json
```

Эквивалентные обёртки: `bash scripts/run-bench-matrix.sh report` и
`bash scripts/run-bench-matrix.sh plot` (вызывают те же скрипты из
`<case>/scripts/`).

## 6. Публикация

Скрипты барьера и санитайза — копии `methodology/scripts/publish/`
(`check-public.sh`, `sanitize-results.sh`), лежат в `<case>/scripts/`.

```bash
# НА ПУБЛИКАЦИОННОЙ КОПИИ дерева (или сначала сделайте архив/бэкап исходных
# results/): --in-place деструктивен и перезаписывает файлы.
# 1) отчёт (read-only): показывает находки и падает при них
bash scripts/sanitize-results.sh
# 2) перезапись приватных путей/имён/хостов/секретов:
bash scripts/sanitize-results.sh --in-place
# 3) барьер публикации: рекурсивный обход дерева кейса с исключением
#    служебных каталогов (`.git/`, `build/`, `dist/`, `__pycache__/`,
#    `node_modules/`, `*.pyc`), grep по обобщённым паттернам — домашние пути
#    `/home/<user>/`, `/root/`, `/Users/<user>/`, точки монтирования `/mnt/`,
#    приватные IPv4 (RFC1918: 10/8, 172.16/12, 192.168/16), секреты по точному
#    паттерну `sk-[A-Za-z0-9_-]{13,}`, `Authorization: Bearer`,
#    `api_key`/`secret`/`token`; fail-closed exit 2 при находке/ошибке, exit 0
#    при чистоте; guard от таймстампов llama.cpp `H.MM.mmm.uuu`. Имена хостов
#    baseline не покрывает — добавляйте их через PUBLIC_DENY_FILE:
bash scripts/check-public.sh
```

**Сырые `results/` не публикуются без санитайза.** Санитайз с `--in-place`
применяйте к **публикационной копии** (или перед запуском сделайте архив/бэкап
исходных `results/`), чтобы не потерять приватные исходные прогоны. Веса модели
в репозиторий **не включаются**; в публикуемых текстах допускаются только
относительные пути.

## 7. Состав артефактов прогона

Каждый прогон пишет каталог `results/<timestamp>-<profile>-<mode>/`:

| Файл | Содержимое |
| --- | --- |
| `result.json` | статус, ошибка, ступени, телеметрия-максимумы, `clocks_*` |
| `profile.json` | использованный профиль (копия) |
| `command.json` | фактическая команда, окружение, порт, seed, mode/order |
| `server.log` | stdout+stderr сервера |
| `telemetry.csv` | сэмплы каждые 0.5 с |

Неудачные прогоны сохраняются как есть со `status = failed`.

## 8. Известные ожидаемые предупреждения

ЗАПОЛНИТЬ: предупреждения, которые не являются ошибкой (например, отсутствие
прав на смену частот GPU, вытеснение host prompt-кеша при больших KV).

## 9. Диагностика

| Симптом | Вероятная причина | Действие |
| --- | --- | --- |
| `profile not found` | неверное имя профиля | свериться со списком в `profiles/` |
| CUDA OOM | контекст/KV не влезает в VRAM | снизить ctx, KV-квант или ubatch |
| `another llama-server is already running` | занят GPU/порт | запускать по одному |
| Порт занят | конфликт с прод-бэкендом | освободить порт |
