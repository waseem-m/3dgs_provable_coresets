# Contributing

Please open an issue before making a large behavioral change. Keep pull
requests focused and document user-visible CLI changes.

Before submitting a change, run:

```bash
python -m compileall -q gs_coresets
gs-coresets --help
gs-coresets coreset --help
```

GPU validation should use a small scene and a new temporary output directory.
