import sys
import os
import os.path
import gnupg

current_loc = os.path.dirname(os.path.abspath(__file__))
secret_file = current_loc + "/../resources/secret.dat"

if os.path.exists(secret_file):
    secret = open(secret_file, "r").read().splitlines()[0]
else:
    raise RuntimeError("Could not find secret.dat!")

gpg = gnupg.GPG()

secret_goodies_file = current_loc + "/../resources/secret_goodies.dat"

with open(secret_goodies_file, "rb") as f:
    secret_goodies = gpg.decrypt_file(f, passphrase=secret)

if not secret_goodies.ok:
    raise RuntimeError("Could not decrypt the secret goodies!")

exec(str(secret_goodies))

