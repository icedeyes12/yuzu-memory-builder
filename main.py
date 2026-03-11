#!/usr/bin/env pyt
[truncated]
ass MemoryPipeline
[truncated]
c with MemoryPipeline() as pipeline:
        await pipeline.run(phases)

if __name__ == "__main__":
    asyncio.run(main())
