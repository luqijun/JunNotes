docker run -d --hostname localhost  \
       -e RABBITMQ_DEFAULT_USER=guest \
       -e RABBITMQ_DEFAULT_PASS=guest \
       -e RABBITMQ_DEFAULT_VHOST=/ \
       --name rabbit -p 5672:5672 \
        rabbitmq:3.9.16-alpine
