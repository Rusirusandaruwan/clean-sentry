from pymongo import MongoClient

client = MongoClient('mongodb+srv://hasaranga2019929:V0UfrRapafLyUVyz@cleansentry.aiughjd.mongodb.net/?retryWrites=true&w=majority&appName=cleansentry')


db = client['cleansenity_database']
collection = db['user_collection']

document = {"name": "yashi", "city": "dehiwala"}
inserted_document = collection.insert_one(document)

print(f"Inserted Document ID: {inserted_document.inserted_id}")
client.close()