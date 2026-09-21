/** Build the three-slide competition section from the user's local template.
 * Requires locally captured assets; private chat screenshots are not committed.
 * Run with the bundled Node runtime. See CASE_DESIGN.md for inputs.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const root = process.cwd();
const runtime = process.env.CASE_RUNTIME_MODULES || 'C:/Users/86151/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES ||= runtime;
const skill = 'C:/Users/86151/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const require = createRequire(path.join(runtime, '_case_builder.cjs'));
const sharp = require('sharp');
const { Canvas } = require(path.join(runtime, '@oai/artifact-tool/node_modules/skia-canvas'));
const { Presentation, PresentationFile, FileBlob } = await import(pathToFileURL(path.join(runtime, '@oai/artifact-tool/dist/artifact_tool.mjs')));
const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const build = path.join(root, '.cache/case-design-build');
const assets = path.join(build, 'assets');
const out = path.join(root, 'output/case-design');
const template = process.env.CASE_TEMPLATE || 'C:/Users/86151/Desktop/案例设计部分.pptx';
const font = '微软雅黑';
const C = { navy:'#164C78', blue:'#368AC0', mid:'#659ABF', pale:'#EAF5FC', line:'#B9D4E6', ink:'#183D56', grey:'#536E81', white:'#FFFFFF', strip:'#D2EFFB' };
await fs.mkdir(out,{recursive:true});

// The source SVG positions are read from the actual application's unselected,
// unfiltered, 100% complete-graph view. Serialization is split to avoid the
// browser tool's 2,000-item array limit.
const graph = JSON.parse(await fs.readFile(path.join(assets,'graph-snapshot.json'),'utf8'));
if(graph.viewBox !== '0 0 1200 760' || graph.nodes.length!==1339 || graph.edges.length!==3319) throw new Error('Unexpected graph snapshot; verify completeness and 100% view');
const graphSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="4800" height="3040" viewBox="${graph.viewBox}"><title>电子电路课程完整知识图谱，100%视图，无节点名称</title><defs><pattern id="dots" width="19" height="19" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#DCE2E5"/></pattern></defs><rect width="1200" height="760" fill="#FAFBFC"/><rect width="1200" height="760" fill="url(#dots)"/><g>${graph.edges.map(e=>`<line x1="${e.x1}" y1="${e.y1}" x2="${e.x2}" y2="${e.y2}" stroke="${e.stroke}" stroke-width="${e.sw}" opacity="${e.opacity}"/>`).join('')}</g><g>${graph.nodes.map(n=>`<circle transform="${n.transform}" r="${n.r}" fill="${n.fill}" stroke="${n.stroke}" stroke-width="${n.sw}"/>`).join('')}</g></svg>`;
await fs.writeFile(path.join(out,'知识图谱_100%_无标签.svg'),graphSvg);
// Draw directly at 4x resolution. Avoid thousands of SVG opacity layers in
// librsvg while preserving the exact system coordinates and stroke styles.
const graphCanvas=new Canvas(4800,3040), gc=graphCanvas.getContext('2d');
gc.scale(4,4);gc.fillStyle='#FAFBFC';gc.fillRect(0,0,1200,760);
gc.fillStyle='#DCE2E5';
for(let y=1;y<760;y+=19)for(let x=1;x<1200;x+=19){gc.beginPath();gc.arc(x,y,1,0,Math.PI*2);gc.fill();}
for(const e of graph.edges){gc.globalAlpha=Number(e.opacity);gc.strokeStyle=e.stroke;gc.lineWidth=parseFloat(e.sw);gc.beginPath();gc.moveTo(Number(e.x1),Number(e.y1));gc.lineTo(Number(e.x2),Number(e.y2));gc.stroke();}
gc.globalAlpha=1;
for(const n of graph.nodes){const [x,y]=n.transform.match(/[-+\d.e]+/g).map(Number).filter(Number.isFinite);gc.fillStyle=n.fill;gc.strokeStyle=n.stroke;gc.lineWidth=parseFloat(n.sw);gc.beginPath();gc.arc(x,y,Number(n.r),0,Math.PI*2);gc.fill();gc.stroke();}
await fs.writeFile(path.join(out,'知识图谱_100%_无标签_高清.png'),await graphCanvas.toBuffer('png'));
console.log('Graph exported at 4800 x 3040');

// Exact crops of real screenshots; pixels and text are not reconstructed.
const crops = [
 ['qa-full.png','qa.png',294,199,482,180],
 ['recommend-full.png','recommend.png',310,388,185,85],
 ['variant-full.png','variant.png',296,426,477,67],
 ['grading-full.png','grading.png',296,388,477,136],
 ['plan-full.png','plan.png',294,390,482,59],
];
for(const [src,dst,left,top,width,height] of crops) await sharp(path.join(assets,src)).extract({left,top,width,height}).png().toFile(path.join(assets,dst));

const imported = await PresentationFile.importPptx(await FileBlob.load(template));
const proto = imported.toProto();
for(const s of proto.slides){
 s.elements=s.elements.filter(e=>e.name!=='文本框 35' && e.name!=='圆角矩形 8' && (e.bbox?.yEmu??0)>=0);
}
function normalizeFonts(x){
 if(!x || typeof x!=='object') return;
 for(const [k,v] of Object.entries(x)){
  if(k==='typeface' || k==='bulletTypeface') x[k]=font;
  else normalizeFonts(v);
 }
}
normalizeFonts(proto);
const p=Presentation.load(proto);
let seq=0;
function box(s,x,y,w,h,fill=C.white,line='none',lw=0,r=0,name='surface'){
 return s.shapes.add({name:`${name}-${++seq}`,geometry:r?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:lw},...(r?{borderRadius:r}:{})});
}
function text(s,t,x,y,w,h,size=28,color=C.ink,bold=false,align='left'){
 const q=box(s,x,y,w,h,'none');q.text=t;q.text.style={typeface:font,fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};return q;
}
function circle(s,x,y,d,fill,line='none',lw=0){return s.shapes.add({name:`circle-${++seq}`,geometry:'ellipse',position:{left:x,top:y,width:d,height:d},fill,line:{fill:line,width:lw}});}
function line(s,x1,y1,x2,y2,color=C.line,lw=1.5){return s.shapes.add({name:`line-${++seq}`,geometry:'line',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1),verticalFlip:y2<y1},fill:'none',line:{fill:color,width:lw}});}
function connect(s,a,b,from,to,color=C.blue,width=2.2,arrow=true){return s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:color,width},tail:{type:arrow?'triangle':'none',width:'med',length:'med'}});}
async function pic(s,file,x,y,w,h,alt){return s.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function title(s,t){box(s,30,124,17,17,C.white,'#000000',3.5);text(s,t,63,106,1166,61,31,'#000000',true);}
function summary(s,t){box(s,34.06,623.39,1211.87,59.64,C.strip,'none',0,4);text(s,t,55,633,1170,40,28,C.ink,true,'center');}

// Slide 1: six agents, with recommendation and variation training merged.
{
 const s=p.slides.items[0];
 title(s,'面向电类课程理实融合，自研知识与学情驱动的多智能体协同教学平台。');
 box(s,318,176,644,35,C.pale,C.line,1.1,7);
 text(s,'教师  设定教学目标，依据反馈调整教学策略',330,177,620,32,24,C.navy,true,'center');
 const specs=[
  ['01','答题报告智能体','分析作答表现','定位薄弱知识',48,227,C.blue],
  ['02','学习规划智能体','依据学习情况','制定学习路径',484,227,C.navy],
  ['03','知识讲解智能体','组织图文讲解','阐释关键概念',920,227,C.blue],
  ['04','练习训练智能体','题库精准荐题','同构变式巩固',920,473,C.navy],
  ['05','实验助教智能体','支持实验设计','分析实验异常',484,473,C.blue],
  ['06','答疑辅导智能体','回应问题追问','提供针对性指导',48,473,C.navy],
 ];
 const cards=specs.map(a=>box(s,a[4],a[5],312,98,C.white,C.line,1.3,9,'agent'));
 const hub=box(s,431,345,418,115,C.pale,C.blue,2,11,'platform');
 connect(s,cards[0],cards[1],'right','left');connect(s,cards[1],cards[2],'right','left');
 connect(s,cards[2],cards[3],'bottom','top');connect(s,cards[3],cards[4],'left','right');
 connect(s,cards[4],cards[5],'left','right');connect(s,cards[5],cards[0],'top','bottom');
 // Fine inner lines express shared knowledge/evidence, outer arrows the cycle.
 for(let i=0;i<cards.length;i++) connect(s,cards[i],hub,i<3?'bottom':'top',i<3?'top':'bottom','#B6CEDF',1.2,false);
 text(s,'电类课程多智能体协同平台',447,352,386,37,30,C.navy,true,'center');
 text(s,'课程知识与学情反馈',447,390,386,31,25,C.grey,false,'center');
 text(s,'手机与电脑访问',441,425,398,28,24,C.grey,false,'center');
 specs.forEach((a,i)=>{const x=a[4],y=a[5];circle(s,x+17,y+24,48,a[6]);text(s,a[0],x+17,y+26,48,42,26,C.white,true,'center');text(s,a[1],x+78,y+9,225,34,27,C.navy,true);text(s,a[2],x+78,y+44,225,25,24,C.grey);text(s,a[3],x+78,y+70,225,25,24,C.grey);});
 text(s,'学生  完成学习任务，获得个性化反馈',310,585,660,29,24,C.navy,true,'center');
 summary(s,'诊断确定学习起点，反馈调整后续任务，实现教学支持的持续迭代。');
 s.speakerNotes.textFrame.setText('来源：用户提供的宏观框架图及案例报告。题库荐题、同类出题按用户确认合并为练习训练智能体。实验助教按案例教学设计展示，未宣称存在已验证的独立系统入口。');
}
// Slide 2: evidence-grounded extraction stages and actual exported graph.
{
 const s=p.slides.items[1];title(s,'关键机制：课程知识图谱自动化构建');
 text(s,'自动化构建流程',52,174,420,36,28,C.navy,true);
 text(s,'《电子电路基础》课程知识图谱',532,174,710,36,28,C.navy,true);
 line(s,500,178,500,606,C.line,1.2);
 const stages=[
  ['教材解析与章节划分','原文、公式与图表分层组织'],
  ['提炼章节摘要','压缩要点，保留原文依据'],
  ['识别知识实体','提取概念、器件与物理量'],
  ['抽取实体关系','形成“实体—关系—实体”'],
  ['消歧合并与图谱生成','同义归并，证据关联可回溯'],
 ];
 stages.forEach((a,i)=>{const y=222+i*77;circle(s,54,y+7,38,C.blue);text(s,String(i+1).padStart(2,'0'),54,y+8,38,35,24,C.white,true,'center');text(s,a[0],110,y,378,34,27,C.navy,true);text(s,a[1],110,y+36,378,32,24,C.grey);if(i<4) line(s,73,y+49,73,y+75,C.line,2);});
 await pic(s,path.join(out,'知识图谱_100%_无标签_高清.png'),520,215,718,363,'系统完整课程知识图谱，100%缩放，无节点名称；1339个节点和3319条连线');
 const metrics=[['202','章节节点'],['1137','知识实体'],['1268','语义关系']];
 metrics.forEach((m,i)=>{const x=537+i*233;text(s,m[0],x,572,93,39,34,C.navy,true,'right');text(s,m[1],x+104,576,124,32,24,C.grey);});
 summary(s,'将教材内容组织为可追溯的知识网络，为答疑、练习训练与学习规划提供依据。');
 s.speakerNotes.textFrame.setText('来源：系统电子电路知识库 semantic_knowledge_graph.json、hierarchical_graph.py 及前端完整图谱的实际渲染数据。导出时 viewBox=0 0 1200 760（100%），无节点名称、搜索或选中高亮。图含202个章节节点和1137个实体节点，3319条总连线中1268条为概念语义关系，另有1850条实体归属和201条章节层级关系。');
}
// Slide 3: real chat excerpts + generated visual with editable teaching labels.
{
 const s=p.slides.items[2];title(s,'具体应用：多智能体支持个性化学习');
 function panel(x,y,name,sub){box(s,x,y,590,238,C.white,C.line,1.1,6);box(s,x,y,590,43,C.pale);text(s,name,x+16,y+4,160,35,29,C.navy,true);text(s,sub,x+178,y+6,395,32,23,C.grey,false,'right');}
 panel(34,181,'答疑辅导','结合课程知识回应学习问题');
 panel(656,181,'练习训练','荐题、变式与批改相衔接');
 panel(34,443,'知识讲解','电流源的恒流特性');
 panel(656,443,'学习规划','依据错题确定学习重点');
 await pic(s,path.join(assets,'qa.png'),50,232,558,178,'8月30日真实会话：电流源工作原理与核心定义');
 // Three exact screenshot excerpts, with large editable labels beside them.
 text(s,'荐题',673,236,58,31,24,C.navy,true);
 await pic(s,path.join(assets,'recommend.png'),746,227,165,73,'原书荐题：电子电路基础学习指导书第84题');
 text(s,'电流源',949,238,270,31,25,C.ink,true);
 text(s,'输出电流与输出电阻',949,269,270,29,23,C.grey);
 line(s,674,301,1225,301,C.line,0.8);
 text(s,'变式',673,315,58,31,24,C.navy,true);
 await pic(s,path.join(assets,'variant.png'),743,305,479,67,'真实同构变式题节选，保留原参数和文字');
 line(s,674,374,1225,374,C.line,0.8);
 text(s,'批改',673,378,58,31,24,C.navy,true);
 // The title and score are the relevant exact excerpt for compact projection.
 await sharp(path.join(assets,'grading-full.png')).extract({left:296,top:432,width:115,height:21}).png().toFile(path.join(assets,'grading-score.png'));
 await pic(s,path.join(assets,'grading-score.png'),745,381,191,32,'真实批改反馈：100/100');
 text(s,'反馈结果与后续建议',949,377,270,31,23,C.grey);
 // Generated picture is intentionally text-free; every teaching label is native.
 await pic(s,path.join(assets,'current-source-visual.png'),51,492,552,164,'生成的电流源等效模型与输出特性主视觉');
 text(s,'I = Iₛ − U/Rₒ',78,491,230,29,24,C.navy,true);
 text(s,'Iₛ',141,560,45,30,24,C.navy,true);
 text(s,'Rₒ',225,560,50,30,24,C.navy,true);
 text(s,'I',316,490,27,29,24,C.navy,true);
 text(s,'U',577,621,24,30,24,C.navy,true);
 text(s,'理想',516,537,60,29,23,C.blue,true);
 text(s,'实际',516,582,60,29,23,'#579FD0',true);
 text(s,'恒流工作区',367,596,150,27,24,C.grey,false,'center');
 text(s,'输出电压受限，工作区内近似恒流',52,650,551,27,24,C.navy,true,'center');
 await pic(s,path.join(assets,'plan.png'),672,498,558,69,'8月30日真实学习规划：第一阶段与核心目标');
 text(s,'由错题主题形成分阶段学习任务',680,577,542,33,25,C.grey,false,'center');
 [['反馈机制','2–3天'],['三种组态','3–4天'],['旁路电容','2天']].forEach((a,i)=>{const x=680+i*188;box(s,x,618,165,55,C.pale,'none',0,5);text(s,a[0],x,619,165,29,24,C.navy,true,'center');text(s,a[1],x,647,165,24,22,C.grey,false,'center');});
 s.speakerNotes.textFrame.setText('截图来源：8月30日“讲解一下电流源的工作原理”五轮会话，session student-024a45c0-d968-4526-b8c4-db5fe7522519。答疑、荐题、变式、批改和学习规划均为系统真实会话局部截图，未经改写。规划主题为反馈机制、三种BJT组态与旁路电容。知识讲解图为本次使用生图工具新制教学示例，不是历史会话截图；文字与标注为可编辑微软雅黑。实际电流源的输出受输出电压范围和有限输出电阻限制，理想模型适用于相应近似条件。');
}

const candidate=path.join(build,'case-design-candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1.5});await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));
 const l=await p.slides.items[i].export({format:'layout'});await fs.writeFile(path.join(build,`slide-${i+1}.layout.json`),await l.text());
}
console.log(JSON.stringify({candidate,previews:[1,2,3].map(i=>path.join(build,`slide-${i}.png`))}));
if(process.argv.includes('--finalize')){
 const finalPath=path.join(out,'案例设计部分_竞赛成稿.pptx');
 const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
  pythonExecutable:'D:/Anaconda/envs/llm/python.exe',
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:3,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'user_request',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(build,'validation.json')});
 console.log(JSON.stringify(result));
}
