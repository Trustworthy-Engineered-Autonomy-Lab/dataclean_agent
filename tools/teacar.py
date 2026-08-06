import os
import paramiko

KNOWN_HOSTS_PATH = os.path.expanduser("~/.ssh/known_hosts")

HOSTNAME = "teacar2"
USERNAME = "nvidia"
PASSWORD = "nvidia"
CONTROLLER_TARGET_DIR = f"/home/{USERNAME}/iros_ws/controllers/"

JUMP_HOSTNAME = "ece-d4100-w02.ad.ufl.edu"
JUMP_USERNAME = "your uf username"
# Please generate a ssh key and use ssh-copy-id to copy the public key to the jump host
# Adding you uf password here is risky
JUMP_KEY = ""

class TEACar:
    def __init__(self):
        self._jump_client = None
        self._target_client = None

    def __enter__(self):
        self._jump_client = paramiko.SSHClient()
        self._jump_client.load_system_host_keys()
        self._jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self._jump_client.connect(
            hostname=JUMP_HOSTNAME,
            username=JUMP_USERNAME,
            key_filename=JUMP_KEY,
        )

        jump_transport = self._jump_client.get_transport()
        channel = jump_transport.open_channel(
            "direct-tcpip",
            (HOSTNAME, 22),  # destination
            ("", 0),         # source (ignored)
        )

        self._target_client = paramiko.SSHClient()
        self._target_client.load_system_host_keys()
        self._target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        self._target_client.connect(
            hostname=HOSTNAME,
            username=USERNAME,
            password=PASSWORD,
            sock=channel
        )

        self._jump_client.save_host_keys(KNOWN_HOSTS_PATH)
        self._target_client.save_host_keys(KNOWN_HOSTS_PATH)

        return self._target_client

    def __exit__(self, exc_type, exc, tb):
        if self._target_client is not None:
            self._target_client.close()
        if self._jump_client is not None:
            self._jump_client.close()

        return False

