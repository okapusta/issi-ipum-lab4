from sqlalchemy.orm import Session
from sqlalchemy import Engine, select
import numpy as np

from images import Images
# reusable function to insert data into the table
def insert_image(engine: Engine, image_path: str, image_embedding: list[float]):
    with Session(engine) as session:
        # create the image object
        image = Images(image_path=image_path, image_embedding=image_embedding)
        # add the image object to the session
        session.add(image)
        # commit the transaction
        session.commit()

# insert some data into the table
N = 100
for i in range(N):
    image_path = f"image_{i}.jpg"
    image_embedding = np.random.rand(512).tolist()
    insert_image(engine, image_path, image_embedding)

# select first image from the table
with Session(engine) as session:
    image = session.query(Images).first()


# Note: pgvector exposes *distances*, not similarities (lower = more similar).
# To rank, sort ascending by `cosine_distance`. To filter by a similarity threshold,
# use `1 - cosine_distance > threshold`.

# order the K images by ascending cosine distance to the first image (smallest distance = most similar)
def find_k_images(engine: Engine, k: int, original_image: Images) -> list[Images]:
    with Session(engine) as session:
        # execution_options={"prebuffer_rows": True} is used to prebuffer the rows, this is useful when we want to fetch the rows in chunks and return them after session is closed
        result = session.execute(
            select(Images)
            .order_by(Images.image_embedding.cosine_distance(original_image.image_embedding))
            .limit(k),
            execution_options={"prebuffer_rows": True}
        ).scalars().all()
        return result

# find the 10 most similar images to the first image
k = 10
similar_images = find_k_images(engine, k, image)
