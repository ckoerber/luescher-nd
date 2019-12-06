"""Tests if zeta function can be imported
"""
from unittest import TestCase


class ZetaTestCase(TestCase):
    def test_zeta_import(self):
        """Test if three d zeta function can be imported
        """
        try:
            from luescher_nd.zeta.extern.pyzeta import zeta
        except ModuleNotFoundError:
            self.fail("Cannot import luescher_nd.zeta.extern.pyzeta.zeta")
