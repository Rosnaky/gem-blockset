# GEM Blockset

This repository contains GPU enhanced math blocks that increase throughput by taking advantage of parallelism.

## Repository

```
.
|-- models
|   |-- abstract_primitives
|   |-- composites
|   `-- primitives
`-- tests
    |-- composites
    `-- primitives
```

Blocks can be found in the `models/` directory. They are separated into folders which represent the scope of the block.

- `abstract_primitives/`: Simplifies redundant code in primitives
- `primitives/`: Contain core math blocks
- `composites/`: Application agnostic blocks that is simply composed of primitives
- `estimators/`: Blocks that produce mathematically significant conclusions, composed with composites and primitives 
