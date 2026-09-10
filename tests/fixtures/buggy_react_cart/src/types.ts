/**
 * Shared cart types and helpers.
 *
 * Plain TypeScript (no JSX) -- parsed with tree-sitter's `typescript`
 * grammar, separately from CartSummary.tsx's `tsx` grammar. Nothing here
 * is buggy; the seeded bug lives in CartSummary.tsx. This file exists so
 * the fixture exercises real TS-only constructs (interface, type alias,
 * generic function) alongside the JSX ones.
 */

export interface CartItem {
  name: string;
  qty: number;
  unitPrice: number;
}

export type Currency = "USD" | "EUR";

export function sumBy<T>(rows: T[], pick: (row: T) => number): number {
  return rows.reduce((total, row) => total + pick(row), 0);
}

export function cartTotal(items: CartItem[]): number {
  return sumBy(items, (item) => item.qty * item.unitPrice);
}
