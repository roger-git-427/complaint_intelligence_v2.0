from rag.chat import looks_analytical


def test_warehouse_questions_route_to_sql():
    assert looks_analytical("Which companies have the most credit card complaints in Texas?")
    assert looks_analytical("How many complaints were timely?")


def test_narrative_questions_route_to_retrieval():
    long = (
        "Someone stole my identity and opened credit cards. "
        "The bureau will not remove the accounts even after I sent a police report. "
        "This is similar to other complaints about improper use of my report."
    )
    assert not looks_analytical(long)
    assert not looks_analytical("Find similar complaints about unauthorized accounts on my credit")
