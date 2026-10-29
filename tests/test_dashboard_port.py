"""Dashboard binding stays independent of the Isaac Sim runtime."""
import errno
import socket
import unittest

from cra373.demo.demo_app import make_server


class DashboardPortTests(unittest.TestCase):
    def test_falls_back_when_preferred_port_is_occupied(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            preferred = occupied.getsockname()[1]
            with self.assertRaises(OSError) as error:
                make_server("127.0.0.1", preferred)
            self.assertEqual(error.exception.errno, errno.EADDRINUSE)
            server = make_server("127.0.0.1", preferred, fallback_ports=10)
            try:
                self.assertGreater(server.server_address[1], preferred)
            finally:
                server.server_close()

    def test_port_zero_uses_os_selected_port(self):
        server = make_server("127.0.0.1", 0, fallback_ports=10)
        try:
            self.assertGreater(server.server_address[1], 0)
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
