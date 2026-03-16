import asyncio
import numpy as np

async def run_sim(i):
    # Use the formatted string to ensure 'i' is passed correctly
    proc = await asyncio.create_subprocess_exec("python3", "2d_wrap_size.py", f"{i:.2f}")
    await proc.wait()
    proc = await asyncio.create_subprocess_exec("python3", "2d_elect_size.py")
    await proc.wait()
    print(f"Finished simulation for i={i:.2f}")


for i in np.linspace(30, 45, 300):
    asyncio.run(run_sim(i))
print(f"process done")

