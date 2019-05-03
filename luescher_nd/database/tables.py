# pylint: disable=R0903
"""Implementation of project tables
"""
from datetime import datetime

import numpy as np

from sqlalchemy import Column

from sqlalchemy.types import Integer
from sqlalchemy.types import DateTime
from sqlalchemy.types import Float

from sqlalchemy.engine.base import Engine

from sqlalchemy.ext.declarative import declarative_base

from opdens.database.base import BaseEntry

BASE = declarative_base(cls=BaseEntry)


class LongRangeEnergyEntry(BASE):
    """Table which stores information about computed operators."""

    __tablename__ = "long-range-energy"

    id = Column(Integer, primary_key=True)
    n1d = Column(Integer, nullable=False)
    epsilon = Column(Float, nullable=False)
    nstep = Column(Integer, nullable=True)
    mu = Column(Float, nullable=False)
    m = Column(Float, nullable=False)
    gbar = Column(Float, nullable=False)
    E = Column(Float, nullable=False)
    nlevel = Column(Integer, nullable=True)
    date = Column(DateTime, default=datetime.utcnow)

    @property
    def L(self) -> float:  # pylint: disable=C0103
        """Returns length of box edge
        """
        return self.n1d * self.epsilon

    def x(self) -> float:
        r"""Returns $2 mu E L^2 / 4 \pi^2$
        """
        return 2 * self.mu * self.E * self.L ** 2 / 4 * np.pi ** 2


def create_all_tables(engine: Engine):
    """Creates all tables in the database

    **Arguments**
        engine: sqlalchemy.engine.base.Engine
            Database engine.
    """
    BASE.metadata.create_all(engine)


def create_db(database: str, dtype: str = "sqlite"):
    """Establishes connection to database and creates all tables

    **Arguments**
        database: str
            Adress of the database

        dtype: str = "sqlite"
            Type of the database
    """
    from sqlalchemy import create_engine

    engine = create_engine(f"{dtype}:///{database}")
    create_all_tables(engine)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description="Script for creating all tables within database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    PARSER.add_argument("database", type=str, help="Location of the database.")
    PARSER.add_argument(
        "--dtype", "-t", type=str, default="sqlite", help="Type of the database."
    )
    ARGS = PARSER.parse_args()

    create_db(ARGS.database, dtype=ARGS.dtype)
