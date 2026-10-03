import {z} from 'zod';
import {foodAnalysisSchema,foodLabelSchema,foodRecommendationSchema} from '@/lib/food';
import {gemma,profileOf,scoreSchema,readBody,errorResponse} from '@/lib/health';

const requestSchema=z.object({summary:z.string().min(20).max(5000),label:foodLabelSchema,assessment:z.object({scores:scoreSchema,overall:z.number().min(0).max(100)}).passthrough(),profile:z.unknown(),healthReport:z.unknown().optional()});

export async function POST(req:Request){
 const started=Date.now();
 try{
  const body=requestSchema.parse(await readBody(req));
  const profile=profileOf(body.profile);
  const context={healthReportJson:body.healthReport??{healthSummary:body.summary,healthScore:body.assessment.overall,riskScores:body.assessment.scores,profile},extractedFoodLabel:body.label};
  const data=await gemma(`Use the completed health summary and extracted food label below to provide cautious, personalized food guidance. Return JSON with exactly this shape: {"decision":"Yes"|"Occasionally"|"No"|"Unknown","verdict":"Generally suitable"|"Consider in moderation"|"Consider another option"|"Insufficient information","recommendationScore":number,"headline":string,"reason":string,"suggestedPortion":string,"personalizedPoints":string[],"watchouts":string[]}. Directly answer whether this person should have this food, while making clear this is educational guidance rather than medical clearance. Base every nutrition statement on extractedFoodLabel; do not invent values or diagnoses. Connect relevant visible sugar, sodium, calories, protein, fat, saturated fat, fiber and ingredients to the supplied health summary and risk scores. recommendationScore is a transparent 0-100 food-fit indicator, not a medical probability. Use Unknown and score 0 if the label is unreadable. suggestedPortion must use a visible serving size or say "Follow the package serving size". Keep the reason concise and provide specific benefits and cautions. Context: ${JSON.stringify(context)}`,undefined,{maxOutputTokens:1200,operation:'personalized food recommendation'});
  const recommendation=foodRecommendationSchema.parse(data);
  return Response.json({...foodAnalysisSchema.parse({...body.label,...recommendation}),meta:{stage:'recommendation',elapsedMs:Date.now()-started}});
 }catch(error){return errorResponse(error,'personalized food recommendation')}
}
