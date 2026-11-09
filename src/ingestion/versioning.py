"""Versioned collections behind an alias: rebuild the index without downtime.

The API searches the name in COLLECTION_NAME ("wands_products"). With versioning that
name is a Qdrant *alias* pointing at a real collection ("wands_products_v3"):

1. ingest into the next version, "wands_products_v4", while v3 keeps serving;
2. validate v4 (point count, BM25 IDF modifier, smoke queries return results);
3. switch the alias to v4 in one atomic request, so a search never sees a
   half-built index;
4. keep the previous version for rollback, and delete older ones.

    python -m src.ingestion.ingest --versioned                      # build, validate, switch
    python -m src.ingestion.ingest --versioned --replace-collection # first time: the alias
                                                                    # takes over a plain collection's name
    python -m src.ingestion.ingest --rollback                       # point the alias back one version
"""
import re

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation, Modifier

SMOKE_QUERIES = ("coffee table", "area rug", "desk lamp")


class PromotionError(Exception):
    pass


def versioned_name(alias: str, version: int) -> str:
    return f"{alias}_v{version}"


def versions(client: QdrantClient, alias: str) -> dict[int, str]:
    """{version: collection name} for every "<alias>_v<N>" collection."""
    pattern = re.compile(rf"^{re.escape(alias)}_v(\d+)$")
    found = {}
    for c in client.get_collections().collections:
        m = pattern.match(c.name)
        if m:
            found[int(m.group(1))] = c.name
    return dict(sorted(found.items()))


def alias_target(client: QdrantClient, alias: str) -> str | None:
    return next((a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias), None)


def next_version(client: QdrantClient, alias: str) -> int:
    return max(versions(client, alias), default=0) + 1


def validate(client: QdrantClient, collection: str, expected_points: int, embedder,
             sparse_name: str, dense_name: str) -> list[str]:
    """Problems that should stop a promotion; empty when the collection is fit to serve."""
    problems = []
    info = client.get_collection(collection)
    if info.points_count != expected_points:
        problems.append(f"{info.points_count} points, expected {expected_points}")
    sparse = (info.config.params.sparse_vectors or {}).get(sparse_name)
    if sparse is None or sparse.modifier != Modifier.IDF:
        problems.append("sparse index has no IDF modifier")
    for q in SMOKE_QUERIES:
        dense, _ = embedder.embed_query(q)
        if not client.query_points(collection, query=dense, using=dense_name, limit=3).points:
            problems.append(f"smoke query {q!r} returned nothing")
    return problems


def _switch(client: QdrantClient, alias: str, collection: str, current: str | None) -> None:
    ops = []
    if current:
        ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
    ops.append(CreateAliasOperation(create_alias=CreateAlias(collection_name=collection, alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=ops)  # applied together, atomically


def promote(client: QdrantClient, alias: str, collection: str, keep: int = 2,
            replace_collection: bool = False) -> str | None:
    """Point `alias` at `collection`; returns the collection it pointed at before.
    Keeps the newest `keep` versions (never deleting the new or the previous one)."""
    current = alias_target(client, alias)
    if current is None and client.collection_exists(alias):
        # A plain collection already uses the alias's name (an unversioned index).
        if not replace_collection:
            raise PromotionError(
                f"'{alias}' is a plain collection, not an alias. Re-run with --replace-collection "
                f"to delete it and let the alias take its name (searches fail for a moment between the two)."
            )
        logger.warning(f"Deleting plain collection '{alias}' so the alias can use its name.")
        client.delete_collection(alias)
    _switch(client, alias, collection, current)
    logger.info(f"Alias '{alias}' -> '{collection}'" + (f" (was '{current}')" if current else ""))

    protected = {collection, current}
    for _, name in list(versions(client, alias).items())[:-keep] if keep > 0 else []:
        if name not in protected:
            client.delete_collection(name)
            logger.info(f"Deleted old version '{name}'.")
    return current


def rollback(client: QdrantClient, alias: str) -> str:
    """Point the alias at the newest version older than the current one."""
    current = alias_target(client, alias)
    if current is None:
        raise PromotionError(f"'{alias}' is not an alias; nothing to roll back.")
    older = [name for v, name in versions(client, alias).items() if name != current
             and v < int(current.rsplit("_v", 1)[1])]
    if not older:
        raise PromotionError(f"No version older than '{current}' to roll back to.")
    _switch(client, alias, older[-1], current)
    logger.info(f"Rolled back alias '{alias}': '{current}' -> '{older[-1]}'")
    return older[-1]
