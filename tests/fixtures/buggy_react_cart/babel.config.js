// Babel config so Jest can run .ts/.tsx (Jest has no native TypeScript
// or JSX support -- it needs a transform). This is a config file, not a
// dependency: the presets themselves are installed globally in the Node
// sandbox image (amop/sandbox/Dockerfile.node), the same way jest is,
// because containers run with network_mode="none" and there is no
// npm-install path at task time.
//
// Presets are referenced by absolute path rather than bare name: babel
// resolves presets relative to this config file's own directory, which
// is the bind-mounted /workspace -- it does not honor NODE_PATH the way
// node's require() does, so bare "@babel/preset-env" would not resolve
// from here.
const GLOBAL_MODULES = "/usr/local/lib/node_modules";

module.exports = {
  presets: [
    [`${GLOBAL_MODULES}/@babel/preset-env`, { targets: { node: "current" } }],
    `${GLOBAL_MODULES}/@babel/preset-typescript`,
    [`${GLOBAL_MODULES}/@babel/preset-react`, { runtime: "classic" }],
  ],
};
