import logging
import os

import asyncpg
import bcrypt

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "GUI_DATABASE_URL",
    "postgresql://postgres:postgres@postgres:5432/chainlit",
)

_pool: asyncpg.Pool | None = None
_db_initialized = False


#: Схема хранения истории диалогов для слоя данных Chainlit.
#:
#: Взята из документации Chainlit по SQLAlchemyDataLayer и приводится
#: дословно: слой обращается к этим таблицам и колонкам по именам, поэтому
#: отступление от схемы даёт ошибку в рантайме, а не при старте. Если
#: очередная версия Chainlit добавила колонку, её надо дописать и в
#: `CHAINLIT_SCHEMA`, и в `CHAINLIT_MIGRATIONS` (иначе на существующих базах
#: дефект вернётся).
#:
#: Таблица `users` не конфликтует с нашей `gui_users`: первая нужна Chainlit
#: для привязки диалогов к пользователю, вторая хранит пароли для входа.
CHAINLIT_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS users (
        "id" UUID PRIMARY KEY,
        "identifier" TEXT NOT NULL UNIQUE,
        "metadata" JSONB NOT NULL,
        "createdAt" TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS threads (
        "id" UUID PRIMARY KEY,
        "createdAt" TEXT,
        "name" TEXT,
        "userId" UUID,
        "userIdentifier" TEXT,
        "tags" TEXT[],
        "metadata" JSONB,
        FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS steps (
        "id" UUID PRIMARY KEY,
        "name" TEXT NOT NULL,
        "type" TEXT NOT NULL,
        "threadId" UUID NOT NULL,
        "parentId" UUID,
        "streaming" BOOLEAN NOT NULL,
        "waitForAnswer" BOOLEAN,
        "isError" BOOLEAN,
        "metadata" JSONB,
        "tags" TEXT[],
        "input" TEXT,
        "output" TEXT,
        "createdAt" TEXT,
        "command" TEXT,
        "start" TEXT,
        "end" TEXT,
        "generation" JSONB,
        "showInput" TEXT,
        "language" TEXT,
        "indent" INT,
        "defaultOpen" BOOLEAN,
        "autoCollapse" BOOLEAN,
        "modes" JSONB,
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elements (
        "id" UUID PRIMARY KEY,
        "threadId" UUID,
        "type" TEXT,
        "url" TEXT,
        "chainlitKey" TEXT,
        "name" TEXT NOT NULL,
        "display" TEXT,
        "objectKey" TEXT,
        "size" TEXT,
        "page" INT,
        "language" TEXT,
        "forId" UUID,
        "mime" TEXT,
        "props" JSONB,
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedbacks (
        "id" UUID PRIMARY KEY,
        "forId" UUID NOT NULL,
        "threadId" UUID NOT NULL,
        "value" INT NOT NULL,
        "comment" TEXT,
        FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
    )
    """,
)


#: Колонки, которые могли добавиться в новых версиях Chainlit после создания
#: таблицы. База могла быть инициализирована старой версией, и
#: ``CREATE TABLE IF NOT EXISTS`` существующую таблицу не меняет, поэтому
#: недостающие колонки добавляются отдельной миграцией. Имя квотится, чтобы
#: иначе Postgres свёл ``autoCollapse`` к ``autocollapse`` (см. issue о
#: пропадающих сообщениях после перезагрузки страницы).
CHAINLIT_MIGRATIONS: tuple[str, ...] = (
    'ALTER TABLE "steps" ADD COLUMN IF NOT EXISTS "autoCollapse" BOOLEAN',
)


async def _apply_migrations(conn):
    for statement in CHAINLIT_MIGRATIONS:
        try:
            await conn.execute(statement)
        except asyncpg.UndefinedTableError:
            # Таблица появится в init_db ниже, миграция применится в пр.
            continue


def async_database_url() -> str:
    """Возвращает строку подключения для асинхронного драйвера.

    Слой данных Chainlit работает через SQLAlchemy и требует в схеме
    подключения ``postgresql+asyncpg``: без этого он падает на попытке
    выполнить запрос асинхронно.

    Returns:
        str: Строка подключения с асинхронным драйвером.
    """
    if DATABASE_URL.startswith("postgresql+asyncpg://"):
        return DATABASE_URL
    return DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


async def _ensure_database():
    db_name = DATABASE_URL.rsplit("/", 1)[-1]
    admin_url = DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
    try:
        conn = await asyncpg.connect(admin_url)
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info("Database created: %s", db_name)
        await conn.close()
    except asyncpg.DuplicateDatabaseError:
        logger.info("Database %s already exists", db_name)


async def init_db():
    global _db_initialized
    if _db_initialized:
        return
    await _ensure_database()
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS gui_users (
                username VARCHAR(255) PRIMARY KEY,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'user',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for statement in CHAINLIT_SCHEMA:
            await conn.execute(statement)
        await _apply_migrations(conn)
    finally:
        await conn.close()
    _db_initialized = True
    logger.info("база данных готова: gui_users и таблицы истории диалогов")


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        await init_db()
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=4)
    return _pool


async def get_user(username: str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT username, password_hash, role FROM gui_users WHERE username = $1",
            username,
        )
    if row is None:
        return None
    return dict(row)


async def create_user(username: str, password: str, role: str = "user") -> bool:
    await init_db()
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO gui_users (username, password_hash, role) "
                "VALUES ($1, $2, $3)",
                username,
                password_hash,
                role,
            )
        logger.info("User created: %s (role=%s)", username, role)
        return True
    except asyncpg.UniqueViolationError:
        logger.warning("User already exists: %s", username)
        return False


async def update_password(
    username: str, current_password: str, new_password: str
) -> tuple[bool, str]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT password_hash FROM gui_users WHERE username = $1",
            username,
        )
        if row is None:
            return False, "User not found"
        if not bcrypt.checkpw(current_password.encode(), row["password_hash"].encode()):
            return False, "Current password is incorrect"
        if len(new_password) < 4:
            return False, "New password must be at least 4 characters"
        new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        await conn.execute(
            "UPDATE gui_users SET password_hash = $1 WHERE username = $2",
            new_hash,
            username,
        )
        logger.info("Password updated for %s", username)
        return True, "Password updated successfully"


async def seed_admin():
    admin_user = os.getenv("ADMIN_USERNAME", "")
    admin_pass = os.getenv("ADMIN_PASSWORD", "")
    if not admin_user or not admin_pass:
        return
    ok = await create_user(admin_user, admin_pass, role="admin")
    if ok:
        logger.info("Admin user seeded: %s", admin_user)
