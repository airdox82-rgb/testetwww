FROM xxxxrt666/gpt-sovits:latest-cu126

# System-Tools & ffmpeg für Audiohandling
RUN apt-get update && \
    apt-get install -y ffmpeg git nano && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: deutsches Sprachpaket oder zusätzliche Python-Pakete
# RUN pip install some-extra-package
