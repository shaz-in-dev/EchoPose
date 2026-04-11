# Build Stage
FROM rust:1.88 as builder
WORKDIR /app
COPY . .
WORKDIR /app/aggregator
RUN cargo build --release --bin aggregator

# Production Stage
FROM debian:bookworm-slim
WORKDIR /app
# Install OpenSSL required by Axum/Tokio if necessary
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/aggregator /app/aggregator
# Expose only the required dynamic ports (mapped in docker-compose)
CMD ["./aggregator"]
