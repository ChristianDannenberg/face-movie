FROM python:3.12-slim

LABEL org.opencontainers.image.title="Face Movie"
LABEL org.opencontainers.image.description="Face Movie with manual face selection and enhanced face detection"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libegl1 \
        libgles2 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py start.sh face_select_extension.py ./
COPY webapp ./webapp

RUN chmod +x /app/start.sh

VOLUME ["/app/payload/"]

ENV FACE_MOVIE_DELEGATE=cpu \
    PYTHONUNBUFFERED=1

EXPOSE 8080

ENTRYPOINT ["./start.sh"]
CMD []
