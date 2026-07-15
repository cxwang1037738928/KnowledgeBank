/**
 * prompts.js — loader for /prompts/prompts.json, where every LLM prompt lives.
 *
 * Each entry is { prompt, created, function } keyed by name: `prompt` is the
 * template sent to the model, `function` documents where and why it is used.
 * Parsed once per process and cached — edit prompts.json, restart the backend.
 *
 * Substitution replaces only the exact `{key}`s passed in `vars`, never other
 * braces — prompts contain literal JSON examples that a template engine or
 * String.replace pattern would mangle.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const PROMPTS_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)), '..', 'prompts', 'prompts.json');

let _prompts = null;

/**
 * The prompt named `name`, with each `{key}` in `vars` substituted.
 * Throws when the prompt is missing — a silently empty prompt would send the
 * model off doing free-form generation.
 */
export function getPrompt(name, vars = {}) {
  _prompts ??= JSON.parse(fs.readFileSync(PROMPTS_PATH, 'utf-8'));
  const template = _prompts[name]?.prompt;
  if (template === undefined) {
    throw new Error(`prompt "${name}" not found in prompts/prompts.json`);
  }
  return Object.entries(vars).reduce(
    (text, [key, value]) => text.replaceAll(`{${key}}`, String(value)), template);
}
