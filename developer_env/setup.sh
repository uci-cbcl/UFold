#!/bin/bash

# Get the current dir.
if [ -n "$BASH_VERSION" ]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
elif [ -n "$ZSH_VERSION" ]; then
    DIR=${0:a:h}  # https://unix.stackexchange.com/a/115431
else
	echo "Error: Unknown shell; cannot determine path to prot_db repository"
fi
export UFOLD_REPO_DIR="$(dirname $DIR)"

git --git-dir="$UFOLD_REPO_DIR/.git" config --local core.autocrlf input
git --git-dir="$UFOLD_REPO_DIR/.git" config filter.prepare_notebook_for_repository.clean 'developer_env/prepare_notebook_for_repository.py'

alias cd_ufold="cd $UFOLD_REPO_DIR"

DOCKER_BASH_HISTORY="$UFOLD_REPO_DIR/data/docker.bash_history"
touch $DOCKER_BASH_HISTORY

DOCKER_IMAGE="ufold"

# docker aliases
alias ufold_docker_build="docker build -t $DOCKER_IMAGE $UFOLD_REPO_DIR/developer_env"

alias ufold_docker_build_mac_m1="docker build --platform linux/amd64 -t $DOCKER_IMAGE $UFOLD_REPO_DIR/developer_env"

alias ufold_docker_run="docker run -it --rm \
    -v $UFOLD_REPO_DIR:/UFold \
    -v $HOME/.config/gcloud:/root/.config/gcloud \
    -v $UFOLD_REPO_DIR/data/docker.bash_history:/root/.bash_history \
    $DOCKER_IMAGE"

alias ufold_docker_jupyter="docker run -it --rm \
    --hostname localhost \
    -v $UFOLD_REPO_DIR:/UFold \
    -v $HOME/.config/gcloud:/root/.config/gcloud \
    -v $UFOLD_REPO_DIR/data/docker.bash_history:/root/.bash_history \
    -p 0.0.0.0:8888:8888 \
    $DOCKER_IMAGE \
    jupyter notebook \
        --port=8888 \
        --ip=0.0.0.0 \
        --allow-root \
        --no-browser \
        --NotebookApp.custom_display_url=http://localhost:8888"

alias ufold_docker_streamlit="docker run -it --rm \
    -v $UFOLD_REPO_DIR:/UFold \
    -v $HOME/.config/gcloud:/root/.config/gcloud \
    -v $UFOLD_REPO_DIR/data/docker.bash_history:/root/.bash_history \
    -p 0.0.0.0:8501:8501 \
    $DOCKER_IMAGE"
