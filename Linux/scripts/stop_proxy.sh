#!/bin/bash
# host_ip=$(cat /etc/resolv.conf |grep "nameserver" |cut -f 2 -d " ")
unset all_proxy
unset http_proxy
unset https_proxy
echo "unset all_proxy http_proxy https_proxy."