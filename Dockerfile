FROM node:20-slim

# Install Python 3 and system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Install Node.js dependencies
COPY package*.json ./
RUN npm install --omit=dev

# Copy project code
COPY . .

# Environment Defaults
ENV PORT=3000
ENV PYTHON_PORT=5000
ENV PYTHON_BIN=python3

EXPOSE 3000

CMD ["node", "server.js"]
