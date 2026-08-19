import os
import re
import threading
from pathlib import PurePosixPath
try:
    import paramiko
except ImportError:  # Configuration/core tests can load physical tool schemas.
    paramiko = None

KNOWN_HOSTS_PATH = os.path.expanduser(os.environ.get("DATACLEAN_KNOWN_HOSTS", "~/.ssh/known_hosts"))
HOST_KEY_POLICY = os.environ.get(
    "DATACLEAN_SSH_HOST_KEY_POLICY", "accept-new"
).strip().lower()
if HOST_KEY_POLICY not in {"accept-new", "strict"}:
    raise ValueError("DATACLEAN_SSH_HOST_KEY_POLICY must be 'accept-new' or 'strict'")
_HOST_KEY_FILE_LOCK = threading.Lock()

HOSTNAME = os.environ.get("DATACLEAN_CAR_HOST", "teacar2")
USERNAME = os.environ.get("DATACLEAN_CAR_USER", "nvidia")
if not re.fullmatch(r"[A-Za-z0-9_-]+", USERNAME):
    raise ValueError("DATACLEAN_CAR_USER contains unsafe characters")
# Preserve the lab car's original connection behavior while allowing a secret
# supplied by the deployment environment to override it. Setting the variable
# to an empty string explicitly disables password authentication.
PASSWORD = os.environ.get("DATACLEAN_CAR_PASSWORD", "nvidia")
def _remote_absolute_path(value, field):
    value = str(value)
    if not PurePosixPath(value).is_absolute() or any(c in value for c in ("\n", "\r", "\0")):
        raise ValueError(f"{field} must be a safe absolute POSIX path")
    return value.rstrip("/") or "/"


IROS_WS_DIR = _remote_absolute_path(
    os.environ.get("DATACLEAN_CAR_WORKSPACE", f"/home/{USERNAME}/iros_ws"),
    "DATACLEAN_CAR_WORKSPACE",
)
CONTROLLER_TARGET_DIR = _remote_absolute_path(os.environ.get(
    "DATACLEAN_CONTROLLER_TARGET_DIR", f"{IROS_WS_DIR}/controllers"
), "DATACLEAN_CONTROLLER_TARGET_DIR")

JUMP_HOSTNAME = os.environ.get("DATACLEAN_JUMP_HOST", "ece-d4100-w02.ad.ufl.edu")
JUMP_USERNAME = os.environ.get("DATACLEAN_JUMP_USER", "jianangu")
JUMP_KEY = os.environ.get("DATACLEAN_JUMP_KEY", "~/.ssh/id_ed25519")


def _ensure_known_hosts_file():
    path = os.path.abspath(KNOWN_HOSTS_PATH)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, mode=0o700, exist_ok=True)
    if not os.path.exists(path):
        fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    return path


_MissingHostKeyPolicyBase = (
    paramiko.MissingHostKeyPolicy if paramiko is not None else object
)


class _AcceptNewHostKeyPolicy(_MissingHostKeyPolicyBase):
    """Persist first-seen keys; Paramiko still rejects changed known keys."""

    def missing_host_key(self, client, hostname, key):
        path = _ensure_known_hosts_file()
        with _HOST_KEY_FILE_LOCK:
            client._host_keys.add(hostname, key.get_name(), key)
            try:
                client.save_host_keys(path)
            except Exception as exc:
                error_type = paramiko.SSHException if paramiko is not None else RuntimeError
                raise error_type(
                    "Accepted the first-seen host key for {!r}, but could not "
                    "persist it to {}: {}".format(hostname, path, exc)
                )


def _ssh_client():
    client = paramiko.SSHClient()
    # Read the normal OpenSSH locations as the original implementation did,
    # plus the explicitly configured task/lab file when it exists.
    client.load_system_host_keys()
    path = os.path.abspath(KNOWN_HOSTS_PATH)
    if os.path.isfile(path):
        client.load_host_keys(path)
    if HOST_KEY_POLICY == "strict":
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
    else:
        client.set_missing_host_key_policy(_AcceptNewHostKeyPolicy())
    return client

class TEACar:
    def __init__(self):
        if paramiko is None:
            raise RuntimeError("Physical-car SSH requires the optional paramiko dependency")
        self._jump_client = None
        self._target_client = None

    def __enter__(self):
        if HOST_KEY_POLICY == "accept-new":
            _ensure_known_hosts_file()
        self._jump_client = _ssh_client()

        connect_kwargs = {
            "hostname": JUMP_HOSTNAME,
            "username": JUMP_USERNAME,
            "timeout": 15,
            "banner_timeout": 15,
            "auth_timeout": 15,
        }
        if JUMP_KEY and os.path.exists(os.path.expanduser(JUMP_KEY)):
            connect_kwargs["key_filename"] = os.path.expanduser(JUMP_KEY)

        self._jump_client.connect(**connect_kwargs)

        jump_transport = self._jump_client.get_transport()
        channel = jump_transport.open_channel(
            "direct-tcpip",
            (HOSTNAME, 22),  # destination
            ("", 0),         # source (ignored)
        )

        self._target_client = _ssh_client()

        target_kwargs = dict(
            hostname=HOSTNAME,
            username=USERNAME,
            sock=channel,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
        )
        if PASSWORD:
            target_kwargs["password"] = PASSWORD
        self._target_client.connect(**target_kwargs)

        return self._target_client

    def __exit__(self, exc_type, exc, tb):
        if self._target_client is not None:
            self._target_client.close()
        if self._jump_client is not None:
            self._jump_client.close()

        return False

    @property
    def jump(self):
        return self._jump_client
