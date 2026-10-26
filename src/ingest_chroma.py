import chroma_db

chroma_client = chroma_db.Client()
collection = chroma_client.create_collection(name='support_tickets')