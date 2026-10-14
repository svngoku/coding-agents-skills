# Deployment Recipes from the Scalingo Tutorials Hub

Condensed, Scalingo-specific steps from the official [Tutorials Hub](https://doc.scalingo.com/tutorials) (checked October 2026). Use these as starting points when a user wants to deploy a well-known tool. Always re-read the linked page before a real deployment: versions, env vars, and sample repos move.

## Table of contents

1. [Status labels](#status-labels)
2. [Patterns shared by the tutorials](#patterns-shared-by-the-tutorials)
3. [Keycloak (Projects + Private Networks)](#keycloak-projects--private-networks)
4. [PostgREST with Keycloak JWT](#postgrest-with-keycloak-jwt)
5. [ModSecurity WAF on the nginx buildpack](#modsecurity-waf-on-the-nginx-buildpack)
6. [WordPress (Bedrock)](#wordpress-bedrock)
7. [n8n](#n8n)
8. [Metabase](#metabase)
9. [Judoscale autoscaling](#judoscale-autoscaling)
10. [Lovable projects](#lovable-projects)
11. [AI demos: Whisper, pgvector, OpenSearch RAG](#ai-demos-whisper-pgvector-opensearch-rag)
12. [Elastic Stack (legacy)](#elastic-stack-legacy)

## Status labels

Each tutorial carries a label. It matters when you recommend one for production:

| Label | Meaning | Pages |
|---|---|---|
| **Tutorial** | Validated by Scalingo | Keycloak, ModSecurity, WordPress, n8n, Metabase |
| **Demo** | Not a supported product; no guarantee it keeps working or suits production | PostgREST, Whisper, pgvector, Lovable, OpenSearch RAG, Elastic Stack |
| **Demo / Partner** | Third-party service with its own billing | Judoscale |

## Patterns shared by the tutorials

- **Sample repo + push.** Most tutorials clone a `github.com/Scalingo/<tool>-scalingo` repo, `scalingo create`, add an addon, and `git push scalingo master` (`main` works too).
- **Terraform variant.** Several pages also show Terraform: `scalingo_app` (`force_https = true`), `scalingo_scm_repo_link` on the fork, `scalingo_addon`, `scalingo_container_type`, then `terraform apply` and a **manual deployment** from the dashboard (Deploy tab → Manual deployment), since creating the link does not deploy. Some older pages pin `stack_id = "scalingo-22"`; use `scalingo-26` for new apps (see `buildpacks.md`).
- **Reference another env var without expanding it locally.** Escape the `$` so the value stays a reference that is resolved at runtime:

  ```bash
  scalingo --app my-app env-set PGRST_SERVER_PORT=\$PORT PGRST_DB_URI=\$SCALINGO_POSTGRESQL_URL
  ```

- **No local persistence.** The filesystem is ephemeral, so tools that default to SQLite or local uploads need a database addon or object storage (n8n → PostgreSQL, WordPress media → S3).
- **Pin versions with env vars.** Buildpacks for packaged tools install the latest tested version on every deploy unless you pin it (`KEYCLOAK_VERSION`, `METABASE_VERSION`, `MODSECURITY_VERSION`, ...). An empty commit plus a push is the usual way to upgrade:

  ```bash
  git commit --allow-empty -m "Upgrade" && git push scalingo master
  ```

## Keycloak (Projects + Private Networks)

<https://doc.scalingo.com/tutorials/keycloak> — clustered Keycloak (Infinispan cache) on a Private Network, published through an nginx reverse proxy. Sample repo: <https://github.com/Scalingo/keycloak-scalingo>.

```bash
scalingo projects-add keycloak
scalingo projects                                   # note the prj-... ID and the private network
scalingo create my-keycloak --project-id <project_id>
scalingo --app my-keycloak addons-add postgresql postgresql-business-1024
scalingo --app my-keycloak env-set BUILDPACK_URL=https://github.com/Scalingo/keycloak-buildpack
scalingo --app my-keycloak scale kc:3:XL
git push scalingo master

# Public reverse proxy in the same project
scalingo create my-nginx --project-id <project_id>
scalingo --app my-nginx env-set KEYCLOAK_PRIVATE_DOMAIN=kc.<private_network_fqdn>
```

```text
kc: /app/keycloak/bin/kc.sh start --optimized
```

- The process type is `kc`, not `web`, so Keycloak is **not** publicly exposed; only the nginx app is. nginx resolves the private domain with resolvers `10.255.0.10 10.255.0.11` and proxies only `/realms/`, `/resources/`, `/.well-known/`.
- Mandatory env: `KC_PROXY_HEADERS=xforwarded`, `KC_HTTP_ENABLED=true`, `KC_HTTP_PORT=8080`, `KC_HOSTNAME`, `KC_CACHE_EMBEDDED_NETWORK_BIND_ADDRESS="match-address:10.240.*"` (cache traffic on the private network), `KC_CACHE_STACK=jdbc-ping`, `KC_CACHE_CONFIG_MUTATE=true`, `KC_BOOTSTRAP_ADMIN_USERNAME`, `KC_BOOTSTRAP_ADMIN_PASSWORD`. Recommended: `KEYCLOAK_VERSION`, `KC_PROXY_TRUSTED_ADDRESSES`.
- Size: at least XL (about 1.5 GB RAM, CPU-heavy); a Business database for redundancy. Health/metrics endpoints are off by default (`KC_HEALTH_ENABLED`, `KC_METRICS_ENABLED`, management port 9000).
- Back up the database before upgrading: Keycloak upgrades can migrate the schema.

## PostgREST with Keycloak JWT

<https://doc.scalingo.com/tutorials/postgrest> (Demo) — PostgreSQL as a REST API with JWT auth and Row-Level Security.

```bash
scalingo create my-postgrest
scalingo --app my-postgrest addons-add postgresql postgresql-starter-512
scalingo --app my-postgrest env-set BUILDPACK_URL=https://raw.githubusercontent.com/Scalingo/scalingo-labs/main/postgrest-buildpack/postgrest-buildpack.tar.gz
scalingo --app my-postgrest env-set POSTGREST_VERSION=<version>
scalingo --app my-postgrest env-set PGRST_DB_URI=\$SCALINGO_POSTGRESQL_URL PGRST_DB_SCHEMAS=api PGRST_SERVER_PORT=\$PORT
scalingo --app my-postgrest pgsql-console          # create schemas, roles, RLS policies
scalingo --app my-postgrest env-set PGRST_JWT_SECRET="$KEYCLOAK_JWKS" PGRST_JWT_AUD=postgrest-api
git commit --allow-empty -m "Deploy PostgREST" && git push scalingo main
```

- `PGRST_JWT_SECRET` holds the Keycloak JWKS from `<keycloak>/realms/<realm>/protocol/openid-connect/certs`.
- Use `FORCE ROW LEVEL SECURITY`: the PostgREST role owns the tables, and owners bypass plain RLS.
- An M container is enough (PostgREST needs about 100–400 MB).

## ModSecurity WAF on the nginx buildpack

<https://doc.scalingo.com/tutorials/modsecurity> — OWASP Core Rule Set in front of an app served by Scalingo's nginx buildpack.

```bash
scalingo --app my-app env-set ENABLE_MODSECURITY=true
git commit --allow-empty -m "Enable ModSecurity" && git push scalingo master
curl -v -X INVALID_HTTP_METHOD https://my-app.osc-fr1.scalingo.io   # expect 403
```

Custom rules live in a repo file referenced from a `location` block of `nginx.conf` / `nginx.conf.erb`. The path **must** start with `/app/`:

```nginx
modsecurity on;
modsecurity_rules_file /app/my_rules.conf;
```

- Disable a rule with `SecRuleRemoveById <id>` at the end of the rules file.
- Every deploy installs the latest ModSecurity and the latest LTS CRS. Pin `MODSECURITY_VERSION` and `MODSECURITY_CORE_RULE_SET_VERSION` to avoid surprise upgrades.
- Logging: `MODSECURITY_DEBUG_LOG_LEVEL` (0–9), `MODSECURITY_AUDIT_LOG_LEVEL` (`On` / `RelevantOnly` / `Off`).
- For IP allowlisting rather than request filtering, use the platform's Application Firewall (`app-firewall-rule-add`, see `cli-reference.md`).

## WordPress (Bedrock)

<https://doc.scalingo.com/tutorials/wordpress> — Bedrock-based WordPress on the PHP buildpack. Sample repo: <https://github.com/Scalingo/wordpress-scalingo>.

```bash
git clone https://github.com/Scalingo/wordpress-scalingo && cd wordpress-scalingo
scalingo create my-wordpress
scalingo --app my-wordpress addons-add mysql mysql-starter-512
scalingo --app my-wordpress scale web:1:XL          # optional
git push scalingo master                            # build log: "Detected Bedrock WordPress"
```

- Mandatory env: `WP_HOME` (app URL), `WP_SITEURL` (app URL + `/wp`), and the eight keys/salts (`AUTH_KEY`, `SECURE_AUTH_KEY`, `LOGGED_IN_KEY`, `NONCE_KEY`, `AUTH_SALT`, `SECURE_AUTH_SALT`, `LOGGED_IN_SALT`, `NONCE_SALT`). `WP_ENV` defaults to `production`.
- **Media must go to object storage**: local uploads are lost on restart. The repo bundles `humanmade/s3-uploads`; set `S3_UPLOADS_BUCKET`, `S3_UPLOADS_KEY`, `S3_UPLOADS_SECRET`, `S3_UPLOADS_REGION` (and optionally `S3_UPLOADS_OBJECT_ACL`). Any S3-compatible storage works.
- Plugins: `composer require --ignore-platform-reqs wpackagist-plugin/<name>`, commit `composer.json` and `composer.lock`. Themes go in `web/app/themes`. Admin is at `/wp/wp-admin`.

## n8n

<https://doc.scalingo.com/tutorials/n8n> — workflow automation. Sample repo: <https://github.com/Scalingo/n8n-scalingo> (Node.js app depending on the `n8n` package).

```bash
git clone https://github.com/Scalingo/n8n-scalingo && cd n8n-scalingo
scalingo create my-n8n
scalingo --app my-n8n addons-add postgresql postgresql-starter-512
scalingo --app my-n8n scale web:1:XL
scalingo --app my-n8n env-set N8N_PORT=\$PORT N8N_PROTOCOL=https DB_TYPE=postgresdb N8N_ENCRYPTION_KEY=<random-string>
git push scalingo master
```

- SQLite (n8n's default) is not usable on Scalingo; PostgreSQL is required.
- Start with XL: n8n is RAM-hungry. Scalingo's own production setup runs queue mode with multi-main: 2 × 2XL `web`, 2 × 2XL `worker`, PostgreSQL Business 2G, Caching Business 1G.
- Upgrade: `npm install n8n@<version> --package-lock-only`, commit `package.json` and `package-lock.json`, push.

## Metabase

<https://doc.scalingo.com/tutorials/metabase> — BI dashboards. Sample repo: <https://github.com/Scalingo/metabase-scalingo>.

```bash
git clone https://github.com/Scalingo/metabase-scalingo && cd metabase-scalingo
scalingo create my-metabase
scalingo --app my-metabase addons-add postgresql postgresql-starter-512   # Metabase's own metadata DB
scalingo --app my-metabase scale web:1:XL
git push scalingo master
```

- No mandatory env vars. Pin with `METABASE_VERSION` (must start with `v`); otherwise each deploy installs the latest.
- Point Metabase at a **copy** of the production database rather than the live one, so analytics queries don't slow the app.

## Judoscale autoscaling

<https://doc.scalingo.com/tutorials/judoscale> (Partner) — autoscale on request or job **queue time** instead of the built-in metrics.

1. Create a Scalingo API token (dashboard → Account → Tokens) and paste it in Judoscale; pick the project and app. The token carries all of its account's rights, so consider a dedicated account.
2. Add the adapter package (Rails example: `judoscale-rails`, `judoscale-sidekiq`).
3. `scalingo --app my-app env-set JUDOSCALE_URL=<url from Judoscale>` (treat it as a secret), commit, push.
4. **Disable the built-in Scalingo Autoscaler** on that process type first (`autoscalers-disable web`): one autoscaler per process type.

Min/max container counts are set in Judoscale; the max is your cost ceiling. Works for `web` (request queue) and workers (job queue).

## Lovable projects

<https://doc.scalingo.com/tutorials/lovable> (Demo) — deploy a Lovable-generated Vite + Nitro + TanStack app.

```ts
// vite.config.ts — keep existing options, add the nitro section
import { defineConfig } from "@lovable.dev/vite-tanstack-config";

export default defineConfig({
  nitro: { preset: "node-server" },
});
```

```text
web: node .output/server/index.mjs
```

```bash
scalingo create my-app
git add vite.config.ts Procfile && git commit -m "Migrate to scalingo"
git push && git push scalingo
```

- Start with an M container. To leave Supabase, restore its backup or dump into a Scalingo PostgreSQL addon (512 MB or more).
- Lovable's GitHub sync can overwrite `Procfile` and `vite.config.ts`; re-check both after each sync before pushing to Scalingo.

## AI demos: Whisper, pgvector, OpenSearch RAG

All three are **Demo** pages. Sample code lives in <https://github.com/Scalingo/scalingo-labs>.

**Speech to text with faster-whisper** (<https://doc.scalingo.com/tutorials/whisper>, folder `whisper-speech-to-text`):

```bash
scalingo create my-whisper
scalingo --app my-whisper env-set MODEL_SIZE=small
git push scalingo main
```

| `MODEL_SIZE` | Container |
|---|---|
| `tiny`, `base` | L |
| `small` (default) | XL or 2XL |
| `medium`, `large`, `turbo` | 3XL |

The model downloads on first boot; poll `/health` until `status` is `ready`. Transcription is synchronous; move it to a worker for long files.

**Recommendations with pgvector** (<https://doc.scalingo.com/tutorials/pgvector-recommendation>, folder `filmrecommendation`):

```bash
scalingo create my-filmreco
scalingo --app my-filmreco addons-add postgresql postgresql-starter-512
git push scalingo main
scalingo --app my-filmreco run python3 setup.py     # enables pgvector, creates and loads the table
```

pgvector is enabled with SQL (`CREATE EXTENSION vector`) from the one-off, not with a database feature toggle. One M `web` container is enough for the demo.

**RAG retrieval with OpenSearch** (<https://doc.scalingo.com/tutorials/opensearch-rag>): uses the bundled k-NN and ml-commons plugins on an OpenSearch addon (start at Starter 1024). Every call is a `curl` against `$SCALINGO_OPENSEARCH_URL` from a one-off:

1. Cluster settings: `plugins.ml_commons.only_run_on_ml_node=false`, `model_access_control_enabled=true`, `native_memory_threshold=99` (dev/test value).
2. Register a model group, then `huggingface/sentence-transformers/all-MiniLM-L6-v2` (`TORCH_SCRIPT`); poll the task for the model ID.
3. Ingest pipeline with a `text_embedding` processor (`passage_text` → `passage_embedding`).
4. Index with `"knn": true`, `default_pipeline`, and a 384-dimension `knn_vector` field.
5. Hybrid search: `neural` + `match` queries combined with `script_score`.

`$SCALINGO_OPENSEARCH_URL` only exists inside the app's containers, so an unquoted `scalingo run curl $SCALINGO_OPENSEARCH_URL...` gets expanded (to nothing) by your local shell. Run the calls from an interactive one-off instead:

```bash
scalingo --app my-app run bash
# inside the container:
curl -s "$SCALINGO_OPENSEARCH_URL/_cluster/health"
```

Vector search is RAM- and CPU-hungry; size the plan to the knowledge base.

## Elastic Stack (legacy)

<https://doc.scalingo.com/tutorials/elastic-stack/overview> (Demo, last updated February 2026) — Logstash + Kibana apps on top of the **Elasticsearch** addon, which Scalingo is phasing out in favour of OpenSearch. Prefer OpenSearch for new projects; use this recipe only to maintain an existing stack.

- Logstash: repo `Scalingo/logstash-scalingo` (branch `es7-compat`), at least one L container, `addons-add elasticsearch elasticsearch-starter-512` (plan as given by the tutorial; confirm with `addons-plans elasticsearch`), mandatory `USER` / `PASSWORD` for HTTP basic auth.
- Ship another app's logs to it with a log drain: `scalingo --app my-app log-drains-add --type elk --url <logstash-url>`.
- Kibana: `BUILDPACK_URL=https://github.com/Scalingo/kibana-buildpack`, `ELASTICSEARCH_URL`, plus `ELASTICSEARCH_TLS_CA_URL` if TLS is forced on the database.
- Curator (index cleanup) runs on the Logstash app through a multi-buildpack (`.buildpacks`: JVM common, Python, Logstash, in that order) and the **Scalingo Scheduler**, which reads a `cron.json` at the repo root:

```json
{
  "jobs": [
    { "command": "0 3,15 * * * curator --config curator.yml log-clean.yml" }
  ]
}
```

Scheduled jobs run as one-off containers; list them with `scalingo --app my-app cron-tasks`.
