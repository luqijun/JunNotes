docker run -d \
  --name postgresql \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v ./postgresql_data:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:latest

