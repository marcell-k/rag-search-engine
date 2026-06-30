import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from psycopg import AsyncConnection


class DatabaseClient:
    def __init__(self) -> None:
        self.conninfo = os.getenv("DATABASE_URL")
        assert isinstance(self.conninfo, str)

        self._pool = AsyncConnectionPool(self.conninfo, open=False)

    async def connect(self) -> None:
        await self._pool.open()

    async def disconnect(self) -> None:
        await self._pool.close()

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[AsyncConnection]:
        async with self._pool.connection() as conn:
            yield conn
