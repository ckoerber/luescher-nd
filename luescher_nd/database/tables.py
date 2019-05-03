# pylint: disable=R0903
"""Implementation of project tables
"""
from typing import Tuple

from datetime import datetime

import numpy as np

from sqlalchemy import Column

from sqlalchemy.types import Integer
from sqlalchemy.types import DateTime
from sqlalchemy.types import Float

from sqlalchemy.engine.base import Engine

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import Session

BASE = declarative_base()


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
    nlevel = Column(Integer, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)

    _keys = {"n1d", "epsilon", "nstep", "mu", "m", "gbar", "E", "nlevel"}

    @property
    def L(self) -> float:  # pylint: disable=C0103
        """Returns length of box edge
        """
        return self.n1d * self.epsilon

    @property
    def x(self) -> float:
        r"""Returns $2 mu E L^2 / 4 \pi^2$
        """
        return 2 * self.mu * self.E * self.L ** 2 / 4 * np.pi ** 2

    @classmethod
    def get_or_create(  # pylint: disable=R0913, R0914
        cls, session: Session, **kwargs
    ) -> Tuple["LongRangeEnergyEntry", bool]:
        """Creates a ConvolutionEntry from a data frame.

        **Arguments**
            session: Session
                The database session for querying and adding entries.

            **kwargs:
                LongRangeEnergyEntry creagion arguments.
        """
        if set(cls._keys) != set(kwargs.keys()):
            raise KeyError(
                "Did not find all keys in arguments for creating `LongRangeEnergyEntry`."
                "\nMissing keys:\n\t"
                "\n\t".join(cls._keys.difference(set(kwargs.keys())))
            )

        instance = session.query(cls).filter_by(**kwargs).first()
        if instance:
            created = False
        else:
            instance = cls(**kwargs)
            session.add(instance)
            created = True

        return instance, created


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
