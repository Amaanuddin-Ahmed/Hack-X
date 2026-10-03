import {foodLabelSchema} from '@/lib/food';
import {gemma,imageSchema,readBody,errorResponse} from '@/lib/health';

export async function POST(req:Request){
 const started=Date.now();
 try{
  const body=await readBody(req),image=imageSchema.parse(body.image);
  const data=await gemma(`Read only the visible food package nutrition label and ingredients. Return JSON with exactly this shape: {"isFood":boolean,"labelDetected":boolean,"product":string,"serving":string,"ingredients":string[],"allergens":string[],"nutrients":[{"name":string,"value":number|null,"unit":string,"dailyValuePercent":number|null,"scale":number}],"confidence":"low"|"medium"|"high","missingFields":string[],"extractionNotes":string[]}. Extract only facts visible in the image. Never infer missing values. Use null when an amount or percent daily value is unreadable. Keep every nutrient on the same visible serving basis. Include calories, carbohydrates, sugar, added sugar, protein, total fat, saturated fat, fiber and sodium when visible. Use these display scales: calories 500 kcal, carbohydrates 75 g, sugar 25 g, added sugar 25 g, protein 30 g, total fat 30 g, saturated fat 15 g, fiber 15 g, sodium 600 mg. For another nutrient choose a sensible positive chart scale. List blurry, cropped, obscured or absent fields in missingFields and explain uncertainty in extractionNotes. If the image is not a readable food label, set labelDetected false, confidence low, product "Unreadable food label", serving "Not readable", empty ingredients/allergens, and one nutrient named "Label readability" with null value, empty unit, null dailyValuePercent and scale 1.`,image,{maxOutputTokens:1400,operation:'food-label extraction'});
  return Response.json({label:foodLabelSchema.parse(data),meta:{stage:'extraction',elapsedMs:Date.now()-started}});
 }catch(error){return errorResponse(error,'food-label extraction')}
}
