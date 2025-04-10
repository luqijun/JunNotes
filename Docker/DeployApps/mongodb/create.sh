docker run -d --restart always \
	--name mongodb \
	-v ./data:/data/ \
	-p 8081:8081 \
	-p 27017:27017 \
    mongo:latest
