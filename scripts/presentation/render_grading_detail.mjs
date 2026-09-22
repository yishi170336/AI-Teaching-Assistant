/** Compact, source-grounded grading evidence graphic. Does not edit the deck. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const runtime=process.env.CASE_RUNTIME_MODULES || 'C:/Users/86151/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require=createRequire(path.join(runtime,'grading-preview.cjs'));
const {Canvas}=require(path.join(runtime,'@oai/artifact-tool/node_modules/skia-canvas'));
const {Presentation}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const attempts=JSON.parse(await fs.readFile('data/practice_attempts.json','utf8'));
const attempt=attempts.find(a=>a.id==='204b113e0e7cb71facac7c0138fed535');
if(!attempt || attempt.grading.score!==100 || !attempt.grading.review.passed ||
 !['1.032','1.18','101.71'].every(v=>attempt.answer.text.includes(v)) ||
 attempt.grading.step_analyses.length!==3 ||
 attempt.grading.step_analyses.some(a=>a.status!=='correct')) {
 throw new Error('The expected source answer/step assessment changed. Review before rendering.');
}
const p=Presentation.create({slideSize:{width:590,height:104}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const font='微软雅黑', navy='#164C78';
const measure=new Canvas(590,104).getContext('2d');
function text(value,x,y,width,height,size,color,bold=false){
 measure.font=`${bold?'bold ':''}${size}px "${font}"`;
 if(measure.measureText(value).width>width) throw new Error(`Text does not fit: ${value}`);
 const q=s.shapes.add({geometry:'rect',position:{left:x,top:y,width,height},fill:'none',line:{fill:'none',width:0}});
 q.text=value;
 q.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none',wrap:'none',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};
}
text('学生答案',10,8,96,27,23,navy,true);
text('I₀(5 V) = 1.032 mA；I₀(20 V) = 1.18 mA',118,8,462,27,22,navy);
text('Rₒ = 101.71 kΩ',118,36,462,27,22,navy);
text('批改结果',10,74,96,27,23,navy,true);
text('100 / 100　公式、计算及推导均正确',118,74,462,27,22,navy);
const out=path.resolve('output/case-design-revision/练习训练_学生答案与批改结果.png');
await fs.mkdir(path.dirname(out),{recursive:true});
const b=await p.export({slide:s,format:'png',scale:3});
await fs.writeFile(out,new Uint8Array(await b.arrayBuffer()));
console.log(out);
