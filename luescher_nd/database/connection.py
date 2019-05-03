"""Interface for establishing connection with a database
"""
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from luescher_nd.utilities import get_logger


class DatabaseSession:
    """Class which allows intuitve acces to databases interfacing sqlalchemy.

    The connection is automatically commited (or rolled back in case of an error)
    and closed.
    """

    def __init__(
        self,
        database: str,
        dtype: str = "sqlite",
        verbose: bool = False,
        commit: bool = True,
    ):
        """Initialize DatabaseConnection for connecting to an sqlite database

        **Arguments**

            database: str
                The database address.

            dtype: str = "sqlite"
                The type of the databse.

            verbose: bool = False
                Return extra output.

            commit: bool = True
                Commit database on exit.

        """
        self.database = database
        self.dtype = dtype
        self.verbose = verbose
        self.commit = commit

        stream_level = logging.INFO if verbose else logging.WARNING
        self.logger = get_logger(stream_level)

        self.session = None
        self.logger.info("Creating engine.")
        self.engine = create_engine(self.connection_string, echo=self.verbose)

    @property
    def connection_string(self):
        """Returns the connection string `{dtype}:///{database}`
        """
        return f"{self.dtype}:///{self.database}"

    def __enter__(self):
        """Establishes a connection to the database and returns the cursor.
        """

        self.logger.info("Binding session.")
        Session = sessionmaker()
        self.session = Session(bind=self.engine)
        return self.session

    def __exit__(self, typ, value, traceback):
        """Commits and closes the database session.

        If no exception occured, the database is commit.
        Else the database is rolled back.
        """
        if traceback is None:
            self.logger.info("Transaction without errors.")
            if self.commit:
                self.logger.info("Committing changes.")
                self.session.commit()
        else:
            self.logger.error("Transaction with errors. Roll back & closing db.")
            self.session.rollback()

        self.logger.info("Closing session.")
        self.session.close()

    def __str__(self):
        """Returns descriptive string
        """
        return f"DatabaseSession({self.database})"

    def __repr__(self):
        """Returns descriptive string
        """
        return str(self)
