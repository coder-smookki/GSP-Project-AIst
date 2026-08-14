# deploy/

Здесь лежат compose-слои, Dockerfile'ы и entry-скрипты развёртывания. Полное
руководство по запуску и эксплуатации перенесено в
[docs/DEPLOY.md](../docs/DEPLOY.md).

Ключевое:

```sh
cd deploy
cp .env.example .env
docker compose up -d   # весь стенд: base + qdrant + orc (obs — отдельным профилем)
```

Краткая карта файлов:

| Файл | Слой | Сервисы |
| --- | --- | --- |
| `docker-compose.yml` | base | `postgres`, `aegra`, `gui`; `include:` qdrant + orc |
| `docker-compose.qdrant.yml` | qdrant | `qdrant` |
| `docker-compose.orchestration.yml` | orc | `prefect-server`, `prefect-runner` |
| `docker-compose.observability.yml` | obs (профиль) | `openlit`, `clickhouse` |
| `Dockerfile` / `Dockerfile.gui` | — | образы aegra и GUI из исходников |