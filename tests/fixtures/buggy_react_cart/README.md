# buggy_react_cart — AMOP fixture repo

A small, self-contained React + TypeScript project used as AMOP's
TypeScript/JSX fixture (Milestone 26, spec Section 19.3). It contains
**one deliberately seeded bug**, with a test suite where one test fails
because of it.

Run its tests directly (from a materialized copy, not this directory):

```
jest
```

Expected on the unfixed baseline: **5 passed, 1 failed** — `renders no
count element at all for an empty cart`.

## The seeded bug

`CartSummary.tsx` guards its count badge with `{items.length && ...}`.
When the cart is empty, `items.length` is `0`, and `0 && ...` evaluates
to `0` rather than `false` — so React renders a literal `0` into the
output:

```
buggy: <div class="cart"><h2>Cart</h2>0<strong class="total">0.00</strong></div>
fixed: <div class="cart"><h2>Cart</h2><strong class="total">0.00</strong></div>
```

The fix is `{items.length > 0 && ...}`.

This is the classic JSX "number in `&&`" leak — a real bug shape React
work actually produces, and one that **cannot be expressed in the Python
or plain-JS fixtures at all**, which is exactly why it was chosen over
another arithmetic typo. Confirmed solvable: applying the one-line fix
takes the suite from 5/6 to 6/6.

## Why the assertion is an exact string match

An empty cart legitimately renders `0.00` as its total, so a loose
`expect(html).not.toContain("0")` would be ambiguous about which zero it
found and could pass for the wrong reason. The failing test asserts the
complete expected markup instead, so the only difference it can report
is the stray `0` itself.

## Two files, two grammars

`src/types.ts` (plain TypeScript: interface, type alias, generic
function) and `src/CartSummary.tsx` (TSX/JSX) are parsed by *different*
tree-sitter grammars — `typescript` and `tsx` respectively — because
`<T>` is genuinely ambiguous between a type assertion and a JSX element.
The fixture deliberately includes both so each grammar path is really
exercised.

## Dependencies and `babel.config.js`

`package.json` declares no dependencies: React, Babel, and Jest are all
installed globally in the Node sandbox image
(`amop/sandbox/Dockerfile.node`), because containers run with
`network_mode="none"` and there is no npm-install path at task time —
the same reason the Python fixtures never declare `pytest`.

`babel.config.js` references its presets by **absolute path**. Babel
resolves presets relative to the config file's own directory (the
bind-mounted `/workspace`) and does *not* honor `NODE_PATH` the way
node's `require()` does, so bare `"@babel/preset-env"` would not resolve.

## Why there is no `.git` directory here

This is tracked as plain files. A nested `.git` inside the AMOP repo
would be stored as a gitlink (mode 160000) and these files would never
actually be committed. Instead, `amop fix` copies this directory into a
per-task scratch dir and runs `git init` + a baseline commit inside the
sandbox container, so every run starts from a byte-identical pristine
repo and nothing here is ever mutated.
