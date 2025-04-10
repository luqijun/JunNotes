docker run -d --restart always \
  --name postgresql \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v ./data:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:latest

