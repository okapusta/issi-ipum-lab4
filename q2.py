from sqlalchemy import select


def find_game(
    engine: Engine,
    game_description: str,
    windows: Optional[bool] = None,
    linux: Optional[bool] = None,
    mac: Optional[bool] = None,
    price: Optional[int] = None
) -> Optional[Games]:
    with Session(engine) as session:
        game_embedding = generate_embeddings(game_description)

        query = (select(Games).order_by(Games.game_description_embedding.cosine_distance(game_embedding)))

        if price:
            query = query.filter(Games.price <= price)
        if windows:
            query = query.filter(Games.windows == True)
        if linux:
            query = query.filter(Games.linux == True)
        if mac:
            query = query.filter(Games.mac == True)

        result = session.execute(query, execution_options={"prebuffer_rows": True})
        game = result.scalars().first()

        return game
