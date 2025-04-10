docker run -d --restart always \
  --name mysql \
  -v ./data:/var/lib/mysql \
  -v ./conf.d:/etc/mysql/conf.d \
  -e MYSQL_ROOT_PASSWORD=123456 \
  -p 3306:3306 \
  mysql:latest

