# pylint: disable=R0903
"""Implementation of project tables
"""
from typing import Tuple
from typing import Set

from datetime import datetime

import numpy as np

from sqlalchemy import Column
from sqlalchemy import ForeignKey

from sqlalchemy.types import Integer
from sqlalchemy.types import DateTime
from sqlalchemy.types import Float
from sqlalchemy.types import String


from sqlalchemy.engine.base import Engine

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import Session


BASE = declarative_base()


class EnergyEntry(BASE):
    """Base implementation of energy table entries
    """

    __tablename__ = "energy"

    id = Column(Integer, primary_key=True)
    n1d = Column(Integer, nullable=False)
    epsilon = Column(Float, nullable=False)
    nstep = Column(Integer, nullable=True)
    mass = Column(Float, nullable=False)
    E = Column(Float, nullable=False)
    nlevel = Column(Integer, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    comment = Column(String, nullable=True)
    discriminator = Column("type", String(50))

    __mapper_args__ = {"polymorphic_on": discriminator}
    _keys: Set[str] = {"n1d", "epsilon", "nstep", "mass", "E", "nlevel"}

    @classmethod
    def keys(cls) -> Set[str]:
        """Returns database keys.
        """
        return cls._keys

    @property
    def L(self) -> float:  # pylint: disable=C0103
        """Returns length of box edge
        """
        return self.n1d * self.epsilon

    def mu(self) -> float:  # pylint: disable=C0103
        """Returns the reduced mass
        """
        return self.mass / 2

    @property
    def x(self) -> float:
        r"""Returns $2 mu E L^2 / 4 \pi^2$
        """
        return 2 * self.mu * self.E * self.L ** 2 / 4 * np.pi ** 2

    @classmethod
    def get_or_create(  # pylint: disable=R0913, R0914
        cls, session: Session, **kwargs
    ) -> Tuple["EnergyEntry", bool]:
        """Creates or returns a EnergyEntry from keywords.

        Arguments:
            session:
                The database session for querying and adding entries.

            kwargs:
                EnergyEntry creagion arguments.
        """
        if cls.keys() != cls.keys().intersection(set(kwargs.keys())):
            raise KeyError(
                "Did not find all keys in arguments for creating `EnergyEntry`."
                "\nMissing keys:\n\t"
                "\n\t".join(cls.keys().difference(set(kwargs.keys())))
            )

        instance = session.query(cls).filter_by(**kwargs).first()
        if instance:
            created = False
        else:
            instance = cls(**kwargs)
            session.add(instance)
            created = True

        return instance, created

    def __repr__(self):
        """Returns string of self"""
        return str(self)

    def __str__(self):
        """Descriptive representation"""
        keys = ", ".join([f"{key}={getattr(self, key)}" for key in self.keys()])
        return f"{self.__class__.__name__}({keys})"


class LongRangeEnergyEntry(EnergyEntry):
    """Table which stores information about computed energy levels for LR potential."""

    __tablename__ = "long-range-energy"

    id = Column(Integer, ForeignKey("energy.id"), primary_key=True)
    gbar = Column(Float, nullable=False)

    _keys = EnergyEntry.keys().union({"gbar"})

    __mapper_args__ = {"polymorphic_identity": "long-range-energy"}


class ContactEnergyEntry(EnergyEntry):
    """
    Table which stores information about computed energy levels for contact potential.
    """

    __tablename__ = "contact-energy"

    id = Column(Integer, ForeignKey("energy.id"), primary_key=True)
    contact_strength = Column(Float, nullable=False)

    _keys = EnergyEntry.keys().union({"contact_strength"})

    __mapper_args__ = {"polymorphic_identity": "contact-energy"}


def create_all_tables(engine: Engine):
    """Creates all tables in the database

    **Arguments**
        engine: sqlalchemy.engine.base.Engine
            Database engine.
    """
    BASE.metadata.create_all(engine)


def create_database(database: str, dtype: str = "sqlite"):
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

    create_database(ARGS.database, dtype=ARGS.dtype)
