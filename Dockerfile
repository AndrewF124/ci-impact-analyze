# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image we labeled every commit in: its Python and SQLite
# decide which sqlglot tests run, so keep it if your image runs them.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS base

FROM base AS pinned
ADD https://codeload.github.com/tobymao/sqlglot/tar.gz/eb11aec4be8cb2122afe713a4816662c3f0fb2fa /tmp/sqlglot.tar.gz
RUN mkdir /pinned && tar -xzf /tmp/sqlglot.tar.gz --strip-components=1 -C /pinned

FROM pinned AS coverage-deps
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.lock

FROM coverage-deps AS coverage
COPY build_coverage.py /build_coverage.py
WORKDIR /pinned
ENV PYTHONPATH=/pinned PYTHONDONTWRITEBYTECODE=1
RUN python /build_coverage.py /coverage.json.gz

FROM base
COPY selector.py patch_parser.py impact.py /app/
COPY --from=pinned /pinned/sqlglot /pinned/sqlglot
COPY --from=coverage /coverage.json.gz /app/coverage.json.gz
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/selector.py"]
