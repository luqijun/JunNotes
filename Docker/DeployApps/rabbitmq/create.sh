docker run -d --restart always \
       -e RABBITMQ_DEFAULT_USER=guest \
       -e RABBITMQ_DEFAULT_PASS=guest \
       -e RABBITMQ_DEFAULT_VHOST=/ \
       -v ./data:/var/lib/rabbitmq \
       --name rabbit \
       -p 15672:15672 \
       -p 5672:5672 \
       rabbitmq:3-management
