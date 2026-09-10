/**
 * A small React cart summary component.
 *
 * Deliberately contains one seeded bug for AMOP's Milestone 26 fixture
 * repo (spec Section 19.3: real code, real failing test, known-good fix
 * so a chain run can be scored pass/fail). The bug is a genuinely
 * JSX-specific one -- not an arithmetic typo -- and cannot be expressed
 * in the Python or plain-JS fixtures at all. Do not "helpfully" fix it
 * by hand; the whole point is that the agent chain finds and fixes it.
 */
import * as React from "react";

import { CartItem, cartTotal } from "./types";

interface CartSummaryProps {
  items: CartItem[];
  label?: string;
}

export function CartSummary({ items, label = "Cart" }: CartSummaryProps) {
  const total = cartTotal(items);
  return (
    <div className="cart">
      <h2>{label}</h2>
      {items.length && <span className="count">{items.length} items</span>}
      <strong className="total">{total.toFixed(2)}</strong>
    </div>
  );
}
