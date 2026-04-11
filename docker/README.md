# Docker Publishing Layout

This folder tracks Docker image release strategy and image contract.

## Planned Images

- `echopose-aggregator` - Rust UDP/WS aggregation service
- `echopose-inference` - Python inference service

## Publish Targets

- GitHub Container Registry (GHCR)
- Docker Hub (optional mirror)

## Notes

The current source Dockerfiles remain in `aggregator/Dockerfile` and `inference/Dockerfile`.
This folder captures release workflow and policy metadata.
