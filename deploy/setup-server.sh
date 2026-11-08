#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 EC2 instance. Run as the default `ubuntu` user:
#
#   ssh ubuntu@<server> 'bash -s' < deploy/setup-server.sh
#
# Safe to run again: each step checks whether it's already done.
set -euo pipefail

REPO_URL="https://github.com/tarun-rai21/ecommerce-rag-engine.git"
BRANCH="${BRANCH:-master}"

# 2 GB of swap: a safety margin for the 2 GB instance during image builds and bursts.
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
fi

# Docker Engine and the Compose plugin from Ubuntu's own repositories.
if ! command -v docker > /dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io docker-compose-v2 git
  sudo systemctl enable --now docker
fi
sudo usermod -aG docker "$USER"  # takes effect at the next login

# Automatic security updates for the OS.
sudo apt-get install -y unattended-upgrades > /dev/null

if [ ! -d ecommerce-rag-engine ]; then
  git clone --branch "$BRANCH" "$REPO_URL"
fi

echo
echo "Done. Next: log in again (for the docker group), create ~/ecommerce-rag-engine/.env"
echo "(see deploy/README section in the main README), then load the index and start the stack."
