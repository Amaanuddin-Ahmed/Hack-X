import {z} from 'zod';

export const nutrientSchema=z.object({
 name:z.string().min(1).max(80),
 value:z.number().nonnegative().nullable(),
 unit:z.string().max(20),
 dailyValuePercent:z.number().min(0).max(1000).nullable(),
 scale:z.number().positive(),
});

export const foodLabelSchema=z.object({
 isFood:z.boolean(),
 labelDetected:z.boolean(),
 product:z.string().min(1).max(200),
 serving:z.string().min(1).max(200),
 ingredients:z.array(z.string().max(150)).max(100).transform(items=>items.slice(0,40)),
 allergens:z.array(z.string().max(100)).max(50).transform(items=>items.slice(0,20)),
 nutrients:z.array(nutrientSchema).min(1).max(100).transform(items=>{
  const seen=new Set<string>();
  return items.filter(item=>{const key=item.name.trim().toLowerCase();if(seen.has(key))return false;seen.add(key);return true}).slice(0,12);
 }),
 confidence:z.enum(['low','medium','high']),
 missingFields:z.array(z.string().max(100)).max(100).transform(items=>items.slice(0,20)),
 extractionNotes:z.array(z.string().max(300)).max(20).transform(items=>items.slice(0,5)),
});

export const foodRecommendationSchema=z.object({
 decision:z.enum(['Yes','Occasionally','No','Unknown']),
 verdict:z.enum(['Generally suitable','Consider in moderation','Consider another option','Insufficient information']),
 recommendationScore:z.number().min(0).max(100),
 headline:z.string().min(1).max(180),
 reason:z.string().min(1).max(1200),
 suggestedPortion:z.string().min(1).max(300),
 personalizedPoints:z.array(z.string().max(500)).min(1).max(6),
 watchouts:z.array(z.string().max(500)).max(6),
});

export const foodAnalysisSchema=foodLabelSchema.extend(foodRecommendationSchema.shape);
export type FoodLabel=z.infer<typeof foodLabelSchema>;
