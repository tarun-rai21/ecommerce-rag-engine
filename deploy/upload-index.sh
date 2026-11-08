#!/usr/bin/env bash
# Copy the local Qdrant collection to the server as a snapshot, so the server never has
# to embed the 43k products itself (about an hour on a laptop CPU, far longer on a t3.small).
#
#   deploy/upload-index.sh ubuntu@<server> [path/to/key.pem]
#
# Needs the local stack running (Qdrant on localhost:6333) and Qdrant running on the
# server (it listens on the server's loopback interface; this script reaches it over SSH).
set -euo pipefail

TARGET="${1:?usage: deploy/upload-index.sh ubuntu@<server> [key.pem]}"
SSH_OPTS=()
[ -n "${2:-}" ] && SSH_OPTS=(-i "$2")
COLLECTION="${COLLECTION:-wands_products}"
LOCAL="http://localhost:6333"

echo "Creating a snapshot of '$COLLECTION' locally..."
NAME=$(curl -fsS -X POST "$LOCAL/collections/$COLLECTION/snapshots?wait=true" \
  | python -c "import json,sys; print(json.load(sys.stdin)['result']['name'])")
FILE="$(mktemp -d)/$NAME"
curl -fsS -o "$FILE" "$LOCAL/collections/$COLLECTION/snapshots/$NAME"
curl -fsS -X DELETE "$LOCAL/collections/$COLLECTION/snapshots/$NAME" > /dev/null  # don't keep it in the volume
echo "Snapshot: $(du -h "$FILE" | cut -f1)"

echo "Copying it to $TARGET..."
scp "${SSH_OPTS[@]}" "$FILE" "$TARGET:/tmp/$NAME"

echo "Restoring it into the server's Qdrant..."
ssh "${SSH_OPTS[@]}" "$TARGET" "curl -fsS -X POST \
  'http://127.0.0.1:6333/collections/$COLLECTION/snapshots/upload?priority=snapshot&wait=true' \
  -F 'snapshot=@/tmp/$NAME' && rm -f '/tmp/$NAME' && echo && \
  curl -fsS 'http://127.0.0.1:6333/collections/$COLLECTION' | python3 -c \
  'import json,sys; print(\"points on the server:\", json.load(sys.stdin)[\"result\"][\"points_count\"])'"
rm -f "$FILE"
