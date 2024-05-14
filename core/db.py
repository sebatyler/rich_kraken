from django.db import connection
from django.db import connections


class ConnectionContextManager:
    def __enter__(self):
        connections.close_all()
        # connections.ensure_defaults()
        # self.conn = connection
        # self.conn.close()
        # self.conn.connect()
        # return self.conn

    def __exit__(self, exc_type, exc_value, traceback):
        connections.close_all()
