#!/bin/bash
# host_ip=$(cat /etc/resolv.conf |grep "nameserver" |cut -f 2 -d " ")
proxy=http://127.0.0.1:7897
export all_proxy=$proxy
export http_proxy=$proxy
export https_proxy=$proxy
export no_proxy="localhost,127.0.0.1,::1"
echo "using proxy $proxy."
