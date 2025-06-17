##############################
#  Dockerfile for Transqlate #
##############################

 # Debian 12 (multi-arch: amd64 & arm64)
FROM python:3.11.7-slim     

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8

# ---- OS-level dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc curl gnupg \
        libpq-dev libsqlite3-dev \
        libmariadb-dev libmariadb-dev-compat \
        unixodbc-dev \
    # --- Microsoft repo: single .deb sets up key + source on all arches ---
    && curl -sSL https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb \
       -o /tmp/msprod.deb \
    && dpkg -i /tmp/msprod.deb && rm /tmp/msprod.deb \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Python dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# ---- Project code & CLI install ----
COPY . .
RUN pip install .

CMD ["transqlate"]