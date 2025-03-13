docker run --name new-api -d --restart always \
       -p 3105:3000 -e TZ=Asia/Shanghai \
       -v ./data:/data calciumion/new-api:latest
