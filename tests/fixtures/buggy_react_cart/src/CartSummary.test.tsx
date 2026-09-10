import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { CartSummary } from "./CartSummary";
import { CartItem, cartTotal, sumBy } from "./types";

const ITEMS: CartItem[] = [
  { name: "Widget", qty: 2, unitPrice: 3.5 },
  { name: "Gadget", qty: 1, unitPrice: 10.0 },
];

// Exact-string assertions rather than `toContain`/`not.toContain`: an
// empty cart legitimately renders "0.00" as its total, so a loose
// "does it contain a 0" check would be ambiguous about WHICH zero it
// found and would pass for the wrong reason.

test("renders the default label", () => {
  const html = renderToStaticMarkup(<CartSummary items={ITEMS} />);
  expect(html).toContain("<h2>Cart</h2>");
});

test("renders a custom label when one is given", () => {
  const html = renderToStaticMarkup(<CartSummary items={ITEMS} label="Basket" />);
  expect(html).toContain("<h2>Basket</h2>");
});

test("renders the item count for a non-empty cart", () => {
  const html = renderToStaticMarkup(<CartSummary items={ITEMS} />);
  expect(html).toContain('<span class="count">2 items</span>');
});

test("renders the cart total", () => {
  const html = renderToStaticMarkup(<CartSummary items={ITEMS} />);
  expect(html).toContain('<strong class="total">17.00</strong>');
});

test("renders no count element at all for an empty cart", () => {
  const html = renderToStaticMarkup(<CartSummary items={[]} />);
  expect(html).toBe(
    '<div class="cart"><h2>Cart</h2><strong class="total">0.00</strong></div>'
  );
});

test("sumBy and cartTotal compute correctly", () => {
  expect(sumBy([1, 2, 3], (n) => n)).toBe(6);
  expect(cartTotal(ITEMS)).toBe(17);
});
