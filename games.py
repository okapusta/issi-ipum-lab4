from typing import List
from sqlalchemy import Integer, Float, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Boolean, Float
from sqlalchemy.engine import Engine
from typing import Optional


from embeddings import generate_embeddings
from images import Base

class Games(Base):
    __tablename__ = "games"
    __table_args__ = {'extend_existing': True}

    # the vector size produced by the model taken from documentation https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
    VECTOR_LENGTH = 512 # check the model output dimensionality

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    # `Text` is an unbounded string in Postgres — game descriptions can run a few KB
    description: Mapped[str] = mapped_column(Text)
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))

from sqlalchemy import select


def find_game(
    engine: Engine,
    game_description: str,
    windows: Optional[bool] = None,
    linux: Optional[bool] = None,
    mac: Optional[bool] = None,
    price: Optional[int] = None,
) -> Optional[Games]:
    with Session(engine) as session:
        game_embedding = generate_embeddings(game_description)

        query = select(Games).order_by(
            Games.game_description_embedding.cosine_distance(game_embedding)
        )

        if price:
            query = query.filter(Games.price <= price)
        if windows:
            query = query.filter(Games.windows == True)
        if linux:
            query = query.filter(Games.linux == True)
        if mac:
            query = query.filter(Games.mac == True)

        result = session.execute(
            query,
            execution_options={"prebuffer_rows": True},
        )
        game = result.scalars().first()

        return game
