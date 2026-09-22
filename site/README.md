# Cross-Sensor Lunar Image Registration — Hugo + Kraiklyn

This site uses the **actual Kraiklyn Hugo theme** from:
https://github.com/jsnjack/kraiklyn

The site content is based on the supplied `Website Details.pdf`. It is a non-interactive research showcase: there are no login pages, forms, or user-data flows.

## 1. Install Kraiklyn

From this project directory:

```bash
./scripts/setup-theme.sh
```

Or manually:

```bash
git clone https://github.com/jsnjack/kraiklyn.git themes/kraiklyn
```

## 2. Run

```bash
hugo server
```

Open `http://localhost:1313/`.

You can also use:

```bash
make server
```

## 3. Build

```bash
hugo
```

or:

```bash
make build
```

The production output is written to `public/`.

## Theme structure

Kraiklyn renders the site's sections on one page. Each section in `content/` has an `anchor` and `weight`, which controls the sidebar link and section order.

No custom theme implementation is included in this project.
