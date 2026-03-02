docker run -it --rm --name n8n -p 5678:5678 --network="host" --env WEBHOOK_URL=https://prova2.frp.darklabs.it --env NODES_EXCLUDE=[] -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
