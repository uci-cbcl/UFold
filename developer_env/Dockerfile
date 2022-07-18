#FROM python:3.9.10
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# install apt-get packages
# set apt-get to non-interactive so it doesn't prompt for configuration (will use the defaults instead)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    htop \
    moreutils \
    screen \
    software-properties-common \
    sqlite3 \
    vim \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install the Google Cloud SDK, which gives us `gcloud`, `gsutil`, `bq`.
# See https://cloud.google.com/sdk/downloads#apt-get
#RUN apt-get update && apt-get -y install apt-transport-https && \
#    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
#    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" \
#        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#    apt-get update && \
#    apt-get install -y \
#    google-cloud-cli \
#    google-cloud-cli-bigtable-emulator \
#    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install seqkit for optimized handling of FASTA files
#RUN wget -P /tmp/ https://github.com/shenwei356/seqkit/releases/download/v2.2.0/seqkit_linux_amd64.tar.gz && \
#    tar -xzvf /tmp/seqkit_linux_amd64.tar.gz && \
#    cp seqkit /usr/local/bin/ && \
#    rm -f /tmp/seqkit_linux_amd64.tar.gz

COPY ./requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm -r ~/.cache/pip

ENV TERM=xterm

# In docker run commands, we mount local repo to /prot_db.
# Make this the default working directory and add to PYTHONPATH.
WORKDIR /UFold
ENV PYTHONPATH="/UFold:$PYTHONPATH"

CMD ["/bin/bash"]
