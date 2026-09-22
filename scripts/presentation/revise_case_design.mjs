/** Targeted revision of the user's manually edited competition deck. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';

const root=process.cwd();
const runtime=process.env.CASE_RUNTIME_MODULES || 'C:/Users/86151/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE_MODULES ||= runtime;
const skill='C:/Users/86151/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const require=createRequire(path.join(runtime,'case_revision.cjs'));
const sharp=require('sharp');
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const build=path.join(root,'.cache/case-design-revision');
const assets=path.join(build,'assets');
const out=path.join(root,'output/case-design-revision');
await fs.mkdir(out,{recursive:true});
const source=process.env.CASE_REVISION_SOURCE || 'C:/Users/86151/Desktop/案例设计部分_竞赛成稿.pptx';
const p0=await PresentationFile.importPptx(await FileBlob.load(source));
const proto=p0.toProto();
// The supplied deck has a transparent, left-aligned title box extending past
// slide 2. Trim only its empty right-hand area; its rendered text stays put.
const graphTitle=proto.slides[1].elements.find(e=>e.id==='27' && e.name==='surface-51');
if(graphTitle) graphTitle.bbox.widthEmu=Math.min(graphTitle.bbox.widthEmu,12192000-graphTitle.bbox.xEmu);
const oldAgentNames=new Set(Array.from({length:30},(_,i)=>i+15).map(i=>([15,20,25,30,35,40].includes(i)?'circle-':'surface-')+i));
proto.slides[0].elements=proto.slides[0].elements.filter(e=>!oldAgentNames.has(e.name) && e.id!=='166');
const oldApplicationNames=new Set(Array.from({length:19},(_,i)=>'surface-'+(111+i)));
proto.slides[2].elements=proto.slides[2].elements.filter(e=>!oldApplicationNames.has(e.name) && !['83','84'].includes(e.id));
const p=Presentation.load(proto);
const C={navy:'#164C78',blue:'#368AC0',line:'#B9D4E6',grey:'#536E81',pale:'#EAF5FC',white:'#FFFFFF'};
let seq=0;
function box(s,x,y,w,h,fill='none',border='none',lw=0,r=0,name='revision'){
 return s.shapes.add({name:`${name}-${++seq}`,geometry:r?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:border,width:lw},...(r?{borderRadius:r}:{})});
}
function text(s,t,x,y,w,h,size=24,color=C.navy,bold=false,align='left'){
 const q=box(s,x,y,w,h);q.text=t;q.text.style={typeface:'微软雅黑',fontSize:size,color,bold,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{left:0,right:0,top:0,bottom:0}};return q;
}
function circle(s,x,y,d,fill='none',border='none',lw=0){return s.shapes.add({name:`icon-circle-${++seq}`,geometry:'ellipse',position:{left:x,top:y,width:d,height:d},fill,line:{fill:border,width:lw}});}
function line(s,x1,y1,x2,y2,color,lw=2){return s.shapes.add({name:`icon-line-${++seq}`,geometry:'line',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1),verticalFlip:y2<y1},fill:'none',line:{fill:color,width:lw}});}
function poly(s,pts,fill,border,lw=2,closed=false){
 const xs=pts.map(a=>a[0]),ys=pts.map(a=>a[1]);const x=Math.min(...xs),y=Math.min(...ys),w=Math.max(...xs)-x,h=Math.max(...ys)-y;
 return s.shapes.add({name:`icon-path-${++seq}`,geometry:'custom',position:{left:x,top:y,width:w,height:h},fill,line:{fill:border,width:lw},customPaths:[{width:w,height:h,commands:[{moveTo:{x:pts[0][0]-x,y:pts[0][1]-y}},...pts.slice(1).map(a=>({lineTo:{x:a[0]-x,y:a[1]-y}})),...(closed?[{close:{}}]:[])]}]});
}
function icon(s,type,x,y,size,color=C.white,bg=C.blue){
 const u=size/32, pt=(a,b)=>[x+a*u,y+b*u];
 const L=(a,b,c,d,w=2)=>line(s,...pt(a,b),...pt(c,d),color,w*u);
 const B=(a,b,c,d,fill='none',w=2,r=0)=>box(s,x+a*u,y+b*u,c*u,d*u,fill,color,w*u,r*u,'icon');
 const O=(a,b,d,fill='none',w=2)=>circle(s,x+a*u,y+b*u,d*u,fill,color,w*u);
 const P=(v,fill='none',closed=false,w=2)=>poly(s,v.map(a=>pt(...a)),fill,color,w*u,closed);
 if(type==='report'){O(3,3,21);L(21,21,30,30,3);B(7,13,3,6,color,0);B(12,9,3,10,color,0);B(17,6,3,13,color,0);}
 if(type==='plan'){P([[7,25],[11,17],[22,17],[26,7]],'none',false,2.3);O(2,21,8,bg);O(9,12,7,bg);O(22,2,8,bg);}
 if(type==='explain'){B(3,3,26,19,'none',2,2);L(16,22,16,28);L(9,29,23,29);P([[7,16],[11,10],[15,15],[20,8],[25,12]],'none',false,2);}
 if(type==='practice'){B(6,5,22,25,'none',2,2);B(11,1,12,6,bg,2,2);P([[10,14],[12,16],[16,11]]);L(19,13,24,13);P([[10,24],[12,26],[16,21]]);L(19,23,24,23);}
 if(type==='lab'){P([[12,3],[12,13],[4,27],[6,30],[26,30],[28,27],[20,13],[20,3]],'none',false,2);L(10,3,22,3);L(8,23,24,23);O(14,15,3,color,0);O(18,20,2,color,0);}
 if(type==='qa'){B(11,12,19,15,'none',2,3);P([[25,27],[29,31],[29,24]],color,true,0);B(2,3,24,18,bg,2,3);P([[6,21],[6,26],[12,21]],color,true,0);[8,14,20].forEach(a=>O(a,10,2.7,color,0));}
 if(type==='teacher'){O(3,3,7,color,0);B(0,13,14,16,color,0,4);B(17,2,14,17,'none',2,1);L(12,16,23,10,2);L(22,20,22,27);L(18,28,28,28);}
 if(type==='student'){P([[1,8],[16,1],[31,8],[16,15]],color,true,0);O(11,13,10,color,0);B(5,23,22,9,color,0,4);L(29,8,29,19,1.6);}
 if(type==='chip'){B(8,8,16,16,color,0,2);for(const a of [10,16,22]){L(a,2,a,7);L(a,25,a,30);L(2,a,7,a);L(25,a,30,a);}}
}
async function pic(s,file,x,y,w,h,alt){return s.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}

// Keep the user's title, summary, central title and all existing connectors.
{
 const s=p.slides.items[0];
 const agents=[
  ['report','答题报告智能体','定位薄弱知识',48,227,'#348FBE'],
  ['plan','学习规划智能体','生成学习路径',484,227,'#5F9571'],
  ['explain','知识讲解智能体','图解关键概念',920,227,'#188D9B'],
  ['practice','练习训练智能体','精准荐题 变式巩固',920,473,'#6976B0'],
  ['lab','实验助教智能体','设计实验 撰写手册',484,473,'#D49836'],
  ['qa','答疑辅导智能体','即时答疑追问',48,473,'#357AAC'],
 ];
 for(const [kind,title,body,x,y,color] of agents){
  circle(s,x+17,y+18,62,color);
  icon(s,kind,x+28,y+29,40,C.white,color);
  text(s,title,x+94,y+16,209,34,27,C.navy,true);
  text(s,body,x+94,y+54,209,30,24,C.grey);
 }
 icon(s,'teacher',363,178,29,C.blue);
 icon(s,'student',400,583,30,'#59805A');
 circle(s,612,353,56,C.white,C.blue,1.5);icon(s,'chip',623,364,34,C.blue);
 s.speakerNotes.textFrame.setText('本次修改仅精简六类智能体的功能说明并补充可编辑图标。保留用户已调整的总起句、总结句与实验助教“设计实验、撰写手册”含义。师生和智能体图标均为原生PowerPoint形状。');
}

if(process.argv.includes('--framework-preview')){
 const b=await p.export({slide:p.slides.items[0],format:'png',scale:1.5});
 await fs.writeFile(path.join(build,'revision-1.png'),new Uint8Array(await b.arrayBuffer()));
 process.exit(0);
}

// The second slide's visible content and geometry are preserved.
{
 const s=p.slides.items[2];
 const subtitle=s.shapes.items.find(e=>e.name==='surface-98');
 if(subtitle){subtitle.text='镜像电流源工作原理';subtitle.text.style={typeface:'微软雅黑',fontSize:23,color:C.grey,alignment:'right',verticalAlignment:'middle',wrap:'none',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};}
 // The user's final selection is the complete generated lecture page.
 // Embed the exact selected PNG, with its full content and original aspect ratio.
 await pic(s,path.join(assets,'mirror-selected-page.png'),45,489,332,187,'用户选定的完整基本镜像电流源讲解页：电路拓扑、镜像原理与线性工作区约束');
 const explanationPoints=[
  ['电路拓扑','标清参考与输出支路'],
  ['镜像机理','关联电路结构与公式'],
  ['工作边界','提示放大区工作条件'],
 ];
 explanationPoints.forEach(([heading,body],i)=>{
  const y=492+i*62;
  text(s,heading,392,y,220,28,25,C.navy,true);
  text(s,body,392,y+29,220,28,24,C.grey);
 });
 await sharp(path.join(build,'mistakes-full.png')).extract({left:295,top:346,width:440,height:53}).png().toFile(path.join(assets,'mistake-knowledge.png'));
 await sharp(path.join(build,'planning-new-full.png')).extract({left:318,top:309,width:638,height:78}).png().toFile(path.join(assets,'planning-mastered.png'));
 text(s,'错题依据',674,491,105,27,24,C.navy,true);
 await pic(s,path.join(assets,'mistake-knowledge.png'),786,492,442,54,'当前系统真实错题本局部：电路故障分析');
 await pic(s,path.join(assets,'planning-mastered.png'),674,552,554,68,'本次在平台重新生成的学习路径：按掌握标准推进，无固定天数');
 const labels=[['反馈判别','标注类型与极性'],['组态比较','按需求选择组态'],['旁路电容','分析故障影响']];
 labels.forEach((a,i)=>{const x=674+i*187;box(s,x,625,180,49,C.pale,'none',0,4);text(s,a[0],x,625,180,26,24,C.navy,true,'center');text(s,a[1],x,650,180,24,22,C.grey,false,'center');});
 s.speakerNotes.textFrame.setText('保留原答疑与练习训练截图。本次知识讲解按用户补充要求改为模拟电子技术中的BJT基本镜像电流源：实际调用平台 KnowledgeExplanationService.generate 的规划、审查、布局与提示词编译流程，在调用平台生图模型之前导出最终提示词，再由Codex内置生图工具绘制。按用户最后确认，知识讲解区域直接嵌入其选定的完整页面，保留原比例及全部内容，完整高清图同时单独交付。对应内容、布局、提示词与运行记录保存在本次制作目录。当前错题本截图真实显示“电路故障分析”，题目为缺少基极直流偏置的共射电路。根据用户要求在平台重新生成不按天数安排的规划，同时保留原会话的反馈、组态与旁路电容主题。底部三项是对新平台规划的可编辑提炼，阶段推进以能否完成指定分析为依据。');
}

const candidate=path.join(build,'revision-candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<3;i++){const b=await p.export({slide:p.slides.items[i],format:'png',scale:1.5});await fs.writeFile(path.join(build,`revision-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));}
if(process.argv.includes('--finalize')){
 const finalPath=path.join(out,process.env.CASE_REVISION_FILENAME || '案例设计部分_竞赛成稿_修订版.pptx');
 const receiptPath=path.join(build,`${path.parse(finalPath).name}-validation.json`);
 const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,pythonExecutable:'D:/Anaconda/envs/llm/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:3,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'user_request',families:['微软雅黑']},verifyArtifactToolImport:true,receiptPath});
 console.log(JSON.stringify(result));
}
console.log('Revision exported.');
