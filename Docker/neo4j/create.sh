docker run \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=./data:/data \
    --name=neo4j -d \
    neo4j

