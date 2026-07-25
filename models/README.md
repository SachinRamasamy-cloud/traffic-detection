# Model files

Model weights are intentionally not committed to Git.

Expected files:

```text
yolo26m.pt               # vehicle detector; can auto-download
license_plate.pt         # custom one-class plate detector
```

The plate model should contain:

```text
0: license_plate
```

Run with:

```bash
traffic-track ... --plate-model models/license_plate.pt
```
