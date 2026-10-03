import {profileOf,scoreSchema,readBody,errorResponse} from '@/lib/health';
export async function POST(req:Request){try{const body=await readBody(req);const profile=profileOf(body.profile);let scores;
 if(body.demo===true)scores={obesity:42,diabetes:36,heart_disease:18};
 else{const endpoint=process.env.MODEL_API_URL||'http://127.0.0.1:8000/predict';const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json',...(process.env.MODEL_API_KEY?{Authorization:`Bearer ${process.env.MODEL_API_KEY}`}:{})},body:JSON.stringify(profile),signal:AbortSignal.timeout(30000)});if(!r.ok)throw Error('The model service could not complete this assessment. Start the local model server and try again.');const output:unknown=await r.json();scores=scoreSchema.parse((output as {scores?:unknown}).scores)}
 const risk=+(Object.values(scores).reduce((a,b)=>a+b,0)/3).toFixed(1);
 return Response.json({scores,overall:+(100-risk).toFixed(1),risk,demo:body.demo===true,bmi:profile.bmi});}catch(e){return errorResponse(e)}}
