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
    finally:
        await conn.close()
    _db_initialized = True
    logger.info("Database initialized (gui_users table ready)")


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
