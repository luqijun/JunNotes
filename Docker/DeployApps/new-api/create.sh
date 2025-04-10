docker run -d --restart always \
       --name new-api \
       -e TZ=Asia/Shanghai \
       -v ./data:/data \
       -p 3105:3000 \
       calciumion/new-api:latest
