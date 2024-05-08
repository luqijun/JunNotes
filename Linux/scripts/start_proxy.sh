#!/bin/bash
# host_ip=$(cat /etc/resolv.conf |grep "nameserver" |cut -f 2 -d " ")
proxy=http://172.30.80.1:9910
export all_proxy=$proxy
export http_proxy=$proxy
export https_proxy=$proxy
echo "using proxy $proxy."

