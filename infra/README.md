# Langfuse local development environment setup

This folder contains the Docker Compose configuration for running Langfuse locally for agent monitoring and tracing.

## Usage

1. Start Langfuse and its database:

```bash
cd infra
docker compose up -d
```

2. Access Langfuse UI at [http://localhost:3000](http://localhost:3000)

3. Use the following environment variables in your agent code to connect to Langfuse:

```
LANGFUSE_API_KEY=dev-api-key
LANGFUSE_SECRET_KEY=dev-secret-key
LANGFUSE_HOST=http://localhost:3000
```

## Notes
- The default credentials and keys are for local development only.
- Data is persisted in a Docker volume (`langfuse-db-data`).
- For more info, see https://langfuse.com/docs/self-hosting
