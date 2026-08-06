# ai_agent_project

Полное руководство по запуску, эксплуатации и диагностике — в [deploy/README.md](deploy/README.md).

Краткая версия:

```sh
cd deploy
cp .env.example .env
docker compose up -d                # базовый стек: postgres + aegra + gui
open http://localhost:8000          # GUI: admin / admin
```

Наблюдаемость LLM (OpenLIT, опционально):

```sh
cd deploy
echo 'OTLP_ENDPOINT=http://openlit:4318' >> .env
docker compose -f docker-compose.yml -f docker-compose.observability.yml --profile obs up -d
open http://localhost:3000          # OpenLIT UI: user@openlit.io / openlituser
```