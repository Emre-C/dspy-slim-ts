/**
 * JSON-serializable shapes for RLM plans, effect payloads, trace extras,
 * and GEPA metric records. Prefer these over `unknown` where values are
 * defined to cross serialization boundaries (plans, LM dict fields, etc.).
 */

export type JsonPrimitive = string | number | boolean | null;

/** Plain JSON object (mutable LM payloads are still assignable here). */
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export type JsonArray = readonly JsonValue[];

export type JsonValue = JsonPrimitive | JsonArray | JsonObject;
