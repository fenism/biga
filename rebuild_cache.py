from signal_cache import SignalCacheBuilder
import time

if __name__ == "__main__":
    builder = SignalCacheBuilder()
    start_time = time.time()
    print(">>> 🚀 Starting Signal Cache Rebuild (Parallel) <<<")
    # Build for the last 600 days to ensure enough data for indicators
    success = builder.build_all_signals()
    
    if success:
        print(f"🎉 Cache rebuild successful! Time taken: {time.time() - start_time:.1f}s")
    else:
        print("❌ Cache rebuild failed.")
