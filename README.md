# genlaw.github.io

## Develop

This website is built into `index.html` from `index.md` and supporting files with `pandoc`. The pandoc command is long, and thus in `build.sh`.

## Build website

```bash
./build.sh
```

## Preview website

```bash
python -m http.server
```

Then open a browser to `localhost:8000`, e.g. on macOS:

```bash
open http://localhost:8000
```