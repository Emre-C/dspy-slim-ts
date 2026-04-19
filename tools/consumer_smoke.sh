#!/usr/bin/env bash
#
# Consumer smoke test for dspy-slim-ts v0.2.0+.
#
# Produces the publish tarball via `npm pack`, installs it into a
# sibling temp directory, and runs both RLM v2 examples end-to-end.
# Fails if any type declaration is unresolvable, any import path is
# broken, or either example exits non-zero.
#
# Release gate: packaged tarball + examples must run clean.
#
# Usage:
#   ./tools/consumer_smoke.sh
#
# Runs in ~10 seconds with the bundled scripted LM; set
# OPENROUTER_API_KEY in the environment to exercise the live path
# (still fast, but consumes a few tokens).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SMOKE_DIR="$(mktemp -d -t dspy-slim-ts-smoke.XXXXXX)"
TARBALL_DIR="$(mktemp -d -t dspy-slim-ts-tgz.XXXXXX)"

trap 'rm -rf "$SMOKE_DIR" "$TARBALL_DIR"' EXIT

echo "[smoke] package dir: $PACKAGE_DIR"
echo "[smoke] smoke dir:   $SMOKE_DIR"

cd "$PACKAGE_DIR"
echo "[smoke] building…"
pnpm build >/dev/null

echo "[smoke] packing…"
npm pack --pack-destination "$TARBALL_DIR" >/dev/null
TARBALL="$(ls "$TARBALL_DIR"/dspy-slim-ts-*.tgz | head -n 1)"
echo "[smoke] tarball: $TARBALL"

cd "$SMOKE_DIR"
echo "[smoke] initialising consumer project…"
cat > package.json <<'EOF'
{
  "name": "dspy-slim-ts-smoke",
  "private": true,
  "version": "0.0.0",
  "type": "module"
}
EOF

cat > tsconfig.json <<'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    "types": ["node"]
  },
  "include": ["quickstart.ts", "custom_effect.ts"]
}
EOF

cp "$PACKAGE_DIR/tools/consumer_smoke_quickstart.ts" quickstart.ts
cp "$PACKAGE_DIR/tools/consumer_smoke_custom_effect.ts" custom_effect.ts

echo "[smoke] installing dependencies…"
npm install "$TARBALL" --silent
npm install --save-dev --silent tsx typescript @types/node

echo "[smoke] typechecking consumer…"
npx tsc

echo "[smoke] running quickstart example…"
npx tsx quickstart.ts

echo
echo "[smoke] running custom-effect example…"
npx tsx custom_effect.ts

echo
echo "[smoke] PASS"
