"""
Monitor the progress of the sampled calculation.
Run this to see real-time updates without blocking.
"""

import time
import os
from pathlib import Path

log_file = 'gravitywavebaseline/sampled_run.log'
results_file = 'gravitywavebaseline/sampled_results.json'

print("="*80)
print("MONITORING SAMPLED CALCULATION")
print("="*80)
print(f"\nLog file: {log_file}")
print(f"Results file: {results_file}")
print("\nPress Ctrl+C to stop monitoring (calculation will continue)")
print("="*80)

last_size = 0
last_update = time.time()

try:
    while True:
        if os.path.exists(log_file):
            current_size = os.path.getsize(log_file)
            
            if current_size > last_size:
                # Read new content
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    if new_content:
                        print(new_content, end='')
                        last_size = current_size
                        last_update = time.time()
        
        # Check if results file exists (calculation complete)
        if os.path.exists(results_file):
            file_age = time.time() - os.path.getmtime(results_file)
            if file_age < 10:  # Recently updated
                print("\n" + "="*80)
                print("[COMPLETE] Results file generated!")
                print("="*80)
                
                import json
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                print(f"\nTests completed: {len(results)}")
                if len(results) > 0:
                    best = results[0]
                    print(f"Best RMS: {best['rms']:.1f} km/s")
                    print(f"Config: {best['config']}")
                    print(f"Period: {best['period_name']}")
                
                break
        
        # Check for timeout (no updates in 5 minutes)
        if time.time() - last_update > 300:
            print("\n[WARN] No updates for 5 minutes - calculation may have stalled")
            break
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\n\n[STOPPED] Monitoring stopped (calculation continues in background)")
    print(f"To see full log: type 'gravitywavebaseline\\sampled_run.log'")

