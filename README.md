# GEM Blockset

This repository contains GPU enhanced math blocks that increase throughput by taking advantage of parallelism.

## Repository

```
.
|-- models
|   `-- primitives
`-- tests
    `-- primitives
```

Blocks can be found in the `models/` directory. They are separated into folders which represent the scope of the block.

- `primitives/`: Contain core math blocks
- `composites/`: Application agnostic blocks that is simply composed of primitives
