docker run -d --restart always \
       --name prometheus \
       -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
       -p 9090:9090 \
       prom/prometheus
