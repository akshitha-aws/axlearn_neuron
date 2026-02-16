#!/bin/bash

# EFA
cd /tmp
sudo rm -rf install_efa 
mkdir install_efa && cd install_efa
# Install EFA Driver (only required for multi-instance training)
curl -s -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz 
wget -q https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key 
cat aws-efa-installer.key | gpg --quiet --fingerprint 
wget -q https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --quiet --verify ./aws-efa-installer-latest.tar.gz.sig 
sudo tar -xf aws-efa-installer-latest.tar.gz 
cd aws-efa-installer && sudo bash efa_installer.sh --yes
cd ..
sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer
sudo rm -rf /tmp/install_efa
cd

# Check
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s  http://169.254.169.254/latest/meta-data/instance-id)
echo "instance_id:$INSTANCE_ID hostname:$(hostname)"
