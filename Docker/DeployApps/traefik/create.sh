docker compose run -d \
       --name traefik \
       -p 80:80 \
       -p 443:443 \
       traefik