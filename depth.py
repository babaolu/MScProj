import asyncio
import numpy as np

async def run_sim(i):
    # Use the formatted string to ensure 'i' is passed correctly
    proc = await asyncio.create_subprocess_exec("python3", "2d_elect_depth.py", f"{i:.2f}")
    await proc.wait()
    print(f"Finished simulation for i={i:.2f}")

r = 42.0
start = np.ceil(r * np.sqrt(0.107))
stop = r * np.sqrt(0.611 + 0.107)

for i in np.linspace(start, stop, 500):
    asyncio.run(run_sim(i))
print(f"process done")

