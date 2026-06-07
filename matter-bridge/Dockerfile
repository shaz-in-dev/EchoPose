FROM node:20-alpine
WORKDIR /app
COPY package.json .
RUN npm install --production --no-audit --no-fund
COPY src/ ./src/
VOLUME ["/app/matter-storage"]
EXPOSE 7788 5540/udp
HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:7788/health || exit 1
CMD ["node", "src/bridge.js"]
