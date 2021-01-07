import random
import sys

if len(sys.argv) > 1:
    rng = random.Random(int(sys.argv[-1]))
else:
    rng = random.Random(0xBA5EBA11)

try:
    while True:
        sys.stdout.write(chr(rng.getrandbits(8)))
except (IOError, KeyboardInterrupt):
    pass
sys.stdout.close()