import {cp, mkdir} from 'node:fs/promises';
const root = new URL('../', import.meta.url), target = new URL('dist/', root);
await mkdir(target, {recursive: true});
// Copy only browser assets, never backend data, model artifacts, test fixtures or historical maps.
for (const name of ['index.html', 'styles.css', 'js']) await cp(new URL(name, root), new URL(name, target), {recursive: true});
console.log('Built static dashboard in evaluator_ui/dist (no dataset or model assets)');
