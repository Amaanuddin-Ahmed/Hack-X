import {z} from 'zod';
import {AppError,imageSchema,profileOf,scoreSchema,readBody,errorResponse} from '@/lib/health';

const xrayResponseSchema=z.object({
 image_type:z.literal('chest_xray'),image_quality:z.string(),status:z.string(),model:z.string(),device:z.enum(['cpu','cuda']),
 model_scores:z.record(z.number()),
 highest_scoring_labels:z.array(z.object({label:z.string(),score:z.number(),confidence:z.string(),note:z.string()})),
 integrated_summary:z.string(),limitations:z.array(z.string()),clinical_review_required:z.boolean(),disclaimer:z.string(),
});

function riskLevel(score:number){return score>=67?'high':score>=34?'moderate':'low'}

export async function POST(req:Request){
 try{
  const body=await readBody(req),profile=profileOf(body.profile),scores=scoreSchema.parse(body.assessment?.scores),image=imageSchema.parse(body.image);
  const prefix=`data:${image.mimeType};base64,`;
  if(!image.data.startsWith(prefix))throw new AppError('Invalid X-ray image encoding.','INVALID_IMAGE',400,false);
  const healthReport={source:'vitalis',overall_health_score:body.assessment?.overall,summary:typeof body.summary==='string'?body.summary.slice(0,5000):null,risk_signals:Object.entries(scores).map(([name,score])=>({model_name:name,label:name.replaceAll('_',' '),risk_level:riskLevel(score),score:score/100})),profile_context:{age:profile.age,gender:profile.gender,bmi:profile.bmi}};
  const modelBase=(process.env.MODEL_API_URL||'http://127.0.0.1:8000/predict').replace(/\/predict\/?$/,'');
  const endpoint=process.env.XRAY_API_URL||`${modelBase}/xray`;
  let response:Response;
  try{response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json',...(process.env.MODEL_API_KEY?{Authorization:`Bearer ${process.env.MODEL_API_KEY}`}:{})},body:JSON.stringify({mime_type:image.mimeType,image_base64:image.data.slice(prefix.length),health_report:healthReport}),signal:AbortSignal.timeout(60000)})}catch(error){if(error instanceof DOMException&&error.name==='TimeoutError')throw new AppError('The X-ray model timed out. Its first run may still be downloading the model checkpoint; retry shortly.','XRAY_TIMEOUT',504,true);throw new AppError('Could not reach the X-ray model service. Start the Python model server and try again.','XRAY_SERVICE_UNAVAILABLE',503,true,error instanceof Error?error.message:undefined)}
  if(!response.ok){let detail='';try{detail=(await response.json() as {detail?:string}).detail||''}catch{}throw new AppError(detail||'The X-ray model could not analyze this image.','XRAY_ANALYSIS_FAILED',response.status>=500?503:response.status,response.status>=500,`Model service HTTP ${response.status}`)}
  const output=xrayResponseSchema.parse(await response.json());
  return Response.json({summary:output.integrated_summary,observations:output.highest_scoring_labels.map(item=>`${item.label}: model score ${item.score.toFixed(3)} (${item.note})`),limitations:output.limitations.join(' '),model:output.model,device:output.device,modelScores:output.model_scores,topFindings:output.highest_scoring_labels,clinicalReviewRequired:output.clinical_review_required,disclaimer:output.disclaimer});
 }catch(error){return errorResponse(error,'chest X-ray analysis')}
}
