// Tiny Vercel Blob CLI for the cloud workers. Needs BLOB_READ_WRITE_TOKEN.
//   node blob.mjs put <local file> <blob path>     -> prints the public URL
//   node blob.mjs list <prefix>                     -> prints "<url> <pathname>" lines
import { put, list } from "@vercel/blob";
import { readFile } from "node:fs/promises";
const [cmd, a1, a2] = process.argv.slice(2);
if (cmd === "put") {
  const r = await put(a2, await readFile(a1), { access: "public", addRandomSuffix: false, allowOverwrite: true,
    multipart: true });
  console.log(r.url);
} else if (cmd === "list") {
  let cursor;
  do {
    const r = await list({ prefix: a1, cursor, limit: 1000 });
    for (const b of r.blobs) console.log(b.url, b.pathname);
    cursor = r.hasMore ? r.cursor : undefined;
  } while (cursor);
} else { console.error("usage: put|list"); process.exit(2); }
