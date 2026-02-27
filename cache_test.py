import time
from signal_cache import SignalCacheBuilder

start = time.time()
print("Init builder...")
builder = SignalCacheBuilder()

def prog(c, t, msg):
    if c == 1 or c % 50 == 0:
        print(f"Prog: {c}/{t} at {time.time() - start:.2f}s")

print("Starting to build signals...")
builder.build_all_signals(progress_callback=prog)
print("Finished at", time.time() - start)
