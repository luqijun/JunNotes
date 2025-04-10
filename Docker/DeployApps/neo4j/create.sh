docker run -d --restart always \
    --name neo4j \
    -p 7474:7474 \
    -p 7687:7687 \
    -v ./data:/data \
    neo4j

