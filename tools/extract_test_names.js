/**
 * extract_test_names.js
 * Extract the first string argument from it(), describe(), and contract().
 * Usage: echo "<source>" | node extract_test_names.js
 * Output: a JSON array such as ["test name 1", "test name 2"]
 */
const acorn = require("acorn");

let source = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (source += chunk));
process.stdin.on("end", () => {
  try {
    const ast = acorn.parse(source, {
      ecmaVersion: 2020,
      sourceType: "script",
    });
    const names = [];
    const targetFns = new Set(["it", "describe", "contract"]);

    function walk(node) {
      if (!node || typeof node !== "object") return;

      if (node.type === "CallExpression") {
        const callee = node.callee;
        // it("name", ...), describe("name", ...), contract("name", ...)
        if (
          callee.type === "Identifier" &&
          targetFns.has(callee.name) &&
          node.arguments.length > 0
        ) {
          const arg = node.arguments[0];
          if (arg.type === "Literal" && typeof arg.value === "string") {
            names.push(arg.value);
          }
          // A template literal without interpolation, such as it(`name`).
          if (
            arg.type === "TemplateLiteral" &&
            arg.quasis.length === 1 &&
            arg.expressions.length === 0
          ) {
            names.push(arg.quasis[0].value.cooked);
          }
        }
      }

      for (const key of Object.keys(node)) {
        const child = node[key];
        if (Array.isArray(child)) {
          child.forEach(walk);
        } else if (typeof child === "object" && child !== null) {
          walk(child);
        }
      }
    }

    walk(ast);
    console.log(JSON.stringify(names));
  } catch (e) {
    console.error("Parse error:", e.message);
    console.log(JSON.stringify([]));
  }
});
